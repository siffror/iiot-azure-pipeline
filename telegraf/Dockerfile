FROM telegraf:1.30
# Copy static config (envs interpoleras av Telegraf där det stöds)
COPY telegraf.conf /etc/telegraf/telegraf.conf
CMD ["--config", "/etc/telegraf/telegraf.conf"]
