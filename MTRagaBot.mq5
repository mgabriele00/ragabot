//+------------------------------------------------------------------+
//|                                    SignalExecutorEA_JAson.mq5    |
//|   Expert Advisor MQL5 che invia dati all'API e gestisce ordini    |
//|   usando il pacchetto JAson.mqh                                  |
//|   TIMEFRAME: 1 Minuto                                           |
//+------------------------------------------------------------------+
#property copyright "© 2025"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <JAson.mqh>    // Copia JAson.mqh in MQL5\Include\

input string InpURL     = "http://127.0.0.1:8081/signal/signal";
input int    InpTimeout = 5000;     // timeout WebRequest (ms)
input uint   InpMagic   = 123456;   // tuo magic number

datetime last_bar_time = 0;
CTrade   trade;

//+------------------------------------------------------------------+
//| OnTick: invocato su ogni nuova chiusura di candela M1           |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime t = iTime(_Symbol, PERIOD_M1, 0);
   if(t == last_bar_time) return;
   last_bar_time = t;
   SendAndExecute();
}

//+------------------------------------------------------------------+
//| Raccoglie dati, invia POST JSON, deserializza con JAson e trade |
//+------------------------------------------------------------------+
void SendAndExecute()
{
   // 1) Preleva fino a 100 barre M1 (minimo 15)
   MqlRates rates[];
   int cnt = CopyRates(_Symbol, PERIOD_M1, 1, 100, rates);
   if(cnt < 15) return;

   // 2) Costruisci JSON "bars"
   string bars = "[";
   for(int i = 0; i < cnt; i++)
      bars += StringFormat(
         "{\"timestamp\":%d,\"open\":%G,\"high\":%G,\"low\":%G,\"close\":%G,\"volume\":%G}%s",
         rates[i].time, rates[i].open, rates[i].high,
         rates[i].low,  rates[i].close, rates[i].tick_volume,
         (i < cnt - 1 ? "," : "")
      );
   bars += "]";

   // 3) Saldo conto
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);

   // 4) Costruisci JSON "open_positions"
   string pos = "[";
   int total = PositionsTotal(), w = 0;
   for(int ix = 0; ix < total; ix++)
   {
      ulong ticket = PositionGetTicket(ix);
      if(ticket == 0 || !PositionSelectByTicket(ticket)) continue;
      ENUM_POSITION_TYPE pt  = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double vol             = PositionGetDouble(POSITION_VOLUME);
      double opn             = PositionGetDouble(POSITION_PRICE_OPEN);
      double slp             = PositionGetDouble(POSITION_SL);
      double tpp             = PositionGetDouble(POSITION_TP);
      string sym             = PositionGetString(POSITION_SYMBOL);

      if(w++ > 0) pos += ",";
      pos += StringFormat(
         "{\"symbol\":\"%s\",\"ticket\":%d,\"type\":\"%s\",\"volume\":%G,"
         "\"open_price\":%G,\"stop_loss\":%s,\"take_profit\":%s}",
         sym, ticket,
         (pt == POSITION_TYPE_BUY ? "buy" : "sell"),
         vol, opn,
         slp > 0 ? DoubleToString(slp, _Digits) : "null",
         tpp > 0 ? DoubleToString(tpp, _Digits) : "null"
      );
   }
   pos += "]";

   // 5) Assemblaggio payload JSON
   string payload = StringFormat(
      "{\"symbol\":\"%s\",\"bars\":%s,\"balance\":%G,\"open_positions\":%s}",
      _Symbol, bars, balance, pos
   );

   // 6) Invia POST via WebRequest
   char   req[];
   int    blen = StringToCharArray(payload, req);
   if(blen <= 0) return;
   ArrayResize(req, blen - 1);  // rimuove il terminatore NUL

   char   resp[];
   string headers = "Content-Type: application/json\r\n";
   string resp_hdr;
   int    status = WebRequest("POST", InpURL, headers, InpTimeout, req, resp, resp_hdr);
   if(status != 200 || ArraySize(resp) == 0)
   {
      PrintFormat("SendAndExecute: HTTP %d, headers=%s", status, resp_hdr);
      return;
   }

   // 7) Deserializza risposta JSON con JAson
   string json = CharArrayToString(resp);
   CJAVal root;
   if(!root.Deserialize(json))
   {
      Print("SendAndExecute: JSON.Deserialize failed");
      return;
   }

   // 8) Esegui chiusure
   if(root.HasKey("orders_to_close"))
   {
      CJAVal closes = root["orders_to_close"];
      for(int i = 0; i < closes.Size(); i++)
      {
         CJAVal o      = closes[i];
         ulong  ticket = (ulong)o["ticket"].ToInt();
         ENUM_POSITION_TYPE pt = POSITION_TYPE_BUY;
         if(PositionSelectByTicket(ticket))
            pt = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         double price = o.HasKey("close_price")
                        ? o["close_price"].ToDbl()
                        : (pt == POSITION_TYPE_BUY
                           ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                           : SymbolInfoDouble(_Symbol, SYMBOL_ASK));
         string comment = o.HasKey("comment") ? o["comment"].ToStr() : "api-close";

         PrintFormat("Close ⇒ ticket=%I64u comm=%s",
                     ticket, comment);

         // full close (CTrade non supporta partial)
         if(!trade.PositionClose(ticket))
            PrintFormat("Close failed ticket=%I64u", ticket);
      }
   }

   // 9) Esegui aperture
   if(root.HasKey("orders_to_open"))
   {
      CJAVal opens = root["orders_to_open"];
      for(int i = 0; i < opens.Size(); i++)
      {
         CJAVal o       = opens[i];
         string sym     = o["symbol"].ToStr();
         string typ     = o["type"].ToStr();
         double volume  = o["volume"].ToDbl();
         double slp     = o.HasKey("stop_loss")   ? o["stop_loss"].ToDbl()   : 0.0;
         double tpp     = o.HasKey("take_profit") ? o["take_profit"].ToDbl() : 0.0;
         string comment = o.HasKey("comment")     ? o["comment"].ToStr()     : "api-open";
         
               // prendo il prezzo di mercato corretto
         double price = (typ == "buy"
                      ? SymbolInfoDouble(sym, SYMBOL_ASK)
                      : SymbolInfoDouble(sym, SYMBOL_BID));

         PrintFormat("Open ⇒ sym=%s type=%s vol=%.5g price=%.5g sl=%.5g tp=%.5g comment=%s",
           sym, typ, volume, price, slp, tpp, comment);

         // Richiesta trade
         MqlTradeRequest req;
         MqlTradeResult  res;
         ZeroMemory(req);
         ZeroMemory(res);

         // Campi obbligatori per evitare retcode 10030
         req.type_time = ORDER_TIME_GTC;
         req.type_filling = ORDER_FILLING_IOC;
         req.action    = TRADE_ACTION_DEAL;
         req.symbol    = sym;
         req.volume    = volume;
         req.type      = (typ == "buy" ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
         req.price     = price;
         req.sl        = slp;
         req.tp        = tpp;
         req.deviation = 10;
         req.magic     = InpMagic;
         req.comment   = comment;

         if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE)
            PrintFormat("Open failed sym=%s ret=%u", sym, res.retcode);
      }
   }
}
//+------------------------------------------------------------------+
