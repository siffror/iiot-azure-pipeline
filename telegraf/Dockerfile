# Baserad på officiell Telegraf-image
FROM telegraf:1.30

# (Valfritt) lägg in en fallback-config — används endast lokalt eller första deploy
# OBS: i ACA kommer riktig config från secret mount!
COPY telegraf.conf /etc/telegraf/telegraf.conf

# Sätt default working directory
WORKDIR /etc/telegraf

# Kör som icke-root (bra säkerhetspraxis)
USER telegraf

# Dokumentera port (valfritt, för läsbarhet)
EXPOSE 8080

# Miljövariabel: Telegraf vet var config ligger
ENV TELEGRAF_CONFIG_PATH=/etc/telegraf/telegraf.conf

# Kör entrypoint (behöver inte CMD-argument längre)
ENTRYPOINT ["telegraf"]
