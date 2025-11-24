#!/bin/bash

# Docker 기반 Elasticsearch 8.8.0 설치 스크립트

echo "🐳 Docker 기반 Elasticsearch 8.8.0 설치를 시작합니다..."

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "❌ Docker가 설치되어 있지 않습니다!"
    echo "   https://www.docker.com/products/docker-desktop 에서 Docker Desktop을 설치하세요."
    exit 1
fi

# Docker 실행 확인
if ! docker info &> /dev/null; then
    echo "❌ Docker가 실행되고 있지 않습니다!"
    echo "   Docker Desktop을 실행하세요."
    exit 1
fi

echo "✅ Docker 확인 완료"

# 기존 컨테이너 정리
echo "🧹 기존 Elasticsearch 컨테이너 정리 중..."
docker stop elasticsearch 2>/dev/null || true
docker rm elasticsearch 2>/dev/null || true

# 비밀번호 생성 (랜덤)
ES_PASSWORD=$(openssl rand -base64 12)
echo "🔑 생성된 비밀번호: $ES_PASSWORD"

# Elasticsearch 컨테이너 실행
echo "🚀 Elasticsearch 컨테이너 시작 중..."
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=true" \
  -e "ELASTIC_PASSWORD=$ES_PASSWORD" \
  -e "xpack.security.http.ssl.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.8.0

# Elasticsearch 시작 대기
echo "⏳ Elasticsearch 시작 대기 중 (60초)..."
sleep 60

# Nori 플러그인 설치
echo "🔧 Nori 한국어 형태소 분석기 설치 중..."
docker exec elasticsearch bin/elasticsearch-plugin install analysis-nori

# 컨테이너 재시작
echo "🔄 컨테이너 재시작 중..."
docker restart elasticsearch
sleep 30

# .env 파일 업데이트
echo "📝 .env 파일 업데이트 중..."
if [ -f ".env" ]; then
    # .env 파일이 있으면 비밀번호만 업데이트
    if grep -q "ELASTICSEARCH_PASSWORD=" .env; then
        sed -i '' "s/ELASTICSEARCH_PASSWORD=.*/ELASTICSEARCH_PASSWORD=$ES_PASSWORD/" .env
    else
        echo "ELASTICSEARCH_PASSWORD=$ES_PASSWORD" >> .env
    fi
else
    # .env 파일이 없으면 생성
    cat > .env << EOF
# Upstage API Configuration
UPSTAGE_API_KEY=your_upstage_api_key_here

# Elasticsearch Configuration
ELASTICSEARCH_PASSWORD=$ES_PASSWORD
EOF
fi

echo ""
echo "✅ 설치 완료!"
echo ""
echo "================================================"
echo "Elasticsearch 비밀번호: $ES_PASSWORD"
echo "================================================"
echo ""
echo "📝 .env 파일에 자동으로 저장되었습니다!"
echo ""
echo "🔍 Elasticsearch 실행 확인:"
echo "   curl -u elastic:$ES_PASSWORD http://localhost:9200"
echo ""
echo "🛑 Elasticsearch 중지:"
echo "   docker stop elasticsearch"
echo ""
echo "🚀 Elasticsearch 재시작:"
echo "   docker start elasticsearch"
echo ""
echo "📊 로그 확인:"
echo "   docker logs elasticsearch"
echo ""
