#!/bin/bash

# Elasticsearch 시작 스크립트

ES_HOME="/Users/dongjunekim/dev_team/ai14/ir/code/elasticsearch-8.8.0"

# Java 경로 설정
export ES_JAVA_HOME="$ES_HOME/jdk"
export JAVA_HOME="$ES_HOME/jdk"

# 보안 설정 비활성화 (개발 환경용)
export xpack.security.enabled=false
export xpack.security.enrollment.enabled=false

# Elasticsearch 시작
echo "Starting Elasticsearch..."
echo "ES_HOME: $ES_HOME"
echo "JAVA_HOME: $JAVA_HOME"

# 기존 프로세스 확인
if pgrep -f elasticsearch > /dev/null; then
    echo "Elasticsearch is already running. Stopping it first..."
    pkill -f elasticsearch
    sleep 2
fi

# 백그라운드에서 실행
cd "$ES_HOME"
./bin/elasticsearch -E xpack.security.enabled=false -E xpack.security.enrollment.enabled=false -d -p /tmp/elasticsearch.pid

# 잠시 대기
sleep 5

# 상태 확인
if curl -s http://localhost:9200 > /dev/null 2>&1; then
    echo "✅ Elasticsearch started successfully on http://localhost:9200"
    curl http://localhost:9200
else
    echo "❌ Failed to start Elasticsearch"
    echo "Checking logs..."
    tail -20 "$ES_HOME/logs/elasticsearch.log"
fi