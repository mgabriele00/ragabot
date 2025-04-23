# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# Dockerfile   MetaTrader 5 on top of accetto/ubuntu‑vnc‑xfce (desktop + VNC/noVNC pre‑built)
# -----------------------------------------------------------------------------
# Build:   docker build -t mt5-vnc .
# Run:     docker run -d --name mt5 \
#                    -e VNC_PASSWORD="mypassword" \
#                    -p 5901:5901 -p 6080:6080 mt5-vnc
# Then:   VNC client → <host>:5901   |   Browser → http://<host>:6080 (noVNC)
# -----------------------------------------------------------------------------

    FROM accetto/ubuntu-vnc-xfce-g3

    USER root
    
    # 1a) Aggiungo architettura i386 e installo Wine (32/64 bit) e dipendenze
    RUN dpkg --add-architecture i386 && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            wine64 wine32 winbind cabextract wget unzip && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

    # 1b) Verifico l'installazione di wineboot
    RUN which wineboot || ls -l /usr/bin/wineboot || echo "wineboot non trovato"

    # 1c) Inizializzo Wine come root usando il percorso completo
    RUN /usr/bin/wineboot --init 
    
    # Passo a utente non‑root preesistente “headless”
    USER headless
    ENV WINEARCH=win32 \
        WINEPREFIX=/home/headless/.wine
    WORKDIR /home/headless
    
    # 2) Creo WINEPREFIX per l'utente headless e installo MetaTrader 5
    RUN mkdir -p $WINEPREFIX && \
        # Potrebbe essere necessario rieseguire wineboot per creare il WINEPREFIX specifico dell'utente
        # wineboot --init && \ 
        wget -q -O mt5setup.exe "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" && \
        wine mt5setup.exe /silent && \
        rm mt5setup.exe
    
    # 3) Auto‑start MetaTrader 5 dentro la sessione VNC/noVNC
    RUN mkdir -p ~/.vnc && \
        printf '%s\n' "#!/bin/bash" \
               "xrdb \$HOME/.Xresources" \
               "startxfce4 &" \
               "sleep 3" \
               'wine "$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe" &' \
               > ~/.vnc/xstartup && \
        chmod +x ~/.vnc/xstartup
    
    # 4) Espongo le porte VNC e noVNC
    EXPOSE 5901 6080
    
    # L'entrypoint originale (supervisord) resta invariato
