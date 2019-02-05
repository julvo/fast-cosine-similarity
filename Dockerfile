FROM docker.elastic.co/elasticsearch/elasticsearch:6.4.1

COPY target/releases/staysense-cosine-sim-6.4.1.zip /data/staysense-cosine-sim-6.4.1.zip

RUN elasticsearch-plugin install file:/data/staysense-cosine-sim-6.4.1.zip

