#!/bin/bash

# Docker로 Elasticsearch 실행 스크립트

echo "🐳 Docker로 Elasticsearch 시작..."

# Docker 실행 확인
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker가 실행되지 않습니다. Docker Desktop을 먼저 실행해주세요."
    echo "   macOS: Docker Desktop 앱을 실행하세요"
    exit 1
fi

# 기존 컨테이너 확인 및 정리
if docker ps -a | grep -q elasticsearch; then
    echo "기존 Elasticsearch 컨테이너 발견. 정리 중..."
    docker stop elasticsearch 2>/dev/null
    docker rm elasticsearch 2>/dev/null
fi

# Elasticsearch 실행 (보안 비활성화, 개발 환경용)
echo "Elasticsearch 컨테이너 시작..."
docker run -d \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "xpack.security.enrollment.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
    docker.elastic.co/elasticsearch/elasticsearch:8.8.0

echo "⏳ Elasticsearch 시작 대기 (약 30초)..."

# 헬스 체크 (최대 30초 대기)
for i in {1..30}; do
    if curl -s http://localhost:9200 > /dev/null 2>&1; then
        echo "✅ Elasticsearch가 성공적으로 시작되었습니다!"
        echo ""
        echo "📊 Elasticsearch 정보:"
        curl -s http://localhost:9200 | python3 -m json.tool
        echo ""
        echo "🔗 접속 URL: http://localhost:9200"
        echo "🔑 보안: 비활성화 (개발 환경)"
        echo ""
        echo "💡 이제 다음 명령어를 실행할 수 있습니다:"
        echo "   python rag_phase3_complete.py"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "❌ Elasticsearch 시작 실패. 로그 확인:"
docker logs elasticsearch --tail 50