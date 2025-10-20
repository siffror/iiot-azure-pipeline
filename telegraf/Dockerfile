FROM telegraf:1.30

# Lägg in statisk konfig
COPY telegraf.conf /etc/telegraf/telegraf.conf

# (valfritt) kör som telegraf-användare; ta bort om du behöver root
USER telegraf

# Dokumentera health-porten (ACA använder inte EXPOSE, men det hjälper läsbarhet)
EXPOSE 8080

ENTRYPOINT ["telegraf"]
CMD ["--config", "/etc/telegraf/telegraf.conf"]
