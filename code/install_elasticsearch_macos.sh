#!/bin/bash

# macOS용 Elasticsearch 8.8.0 설치 스크립트

echo "🚀 Elasticsearch 8.8.0 설치를 시작합니다..."

# 시스템 아키텍처 확인
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "✅ Apple Silicon (M1/M2/M3) 감지됨"
    ES_URL="https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.8.0-darwin-aarch64.tar.gz"
    ES_FILE="elasticsearch-8.8.0-darwin-aarch64.tar.gz"
elif [ "$ARCH" = "x86_64" ]; then
    echo "✅ Intel Mac 감지됨"
    ES_URL="https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.8.0-darwin-x86_64.tar.gz"
    ES_FILE="elasticsearch-8.8.0-darwin-x86_64.tar.gz"
else
    echo "❌ 지원하지 않는 아키텍처: $ARCH"
    exit 1
fi

# 1. Elasticsearch 다운로드
echo "📥 Elasticsearch 다운로드 중..."
if command -v wget &> /dev/null; then
    wget $ES_URL
elif command -v curl &> /dev/null; then
    curl -L -O $ES_URL
else
    echo "❌ wget 또는 curl이 필요합니다. Homebrew로 설치하세요:"
    echo "   brew install wget"
    exit 1
fi

# 2. 압축 해제
echo "📦 압축 해제 중..."
tar -xzf $ES_FILE

# 3. Nori 형태소 분석기 설치
echo "🔧 Nori 한국어 형태소 분석기 설치 중..."
./elasticsearch-8.8.0/bin/elasticsearch-plugin install analysis-nori

# 4. Elasticsearch 데몬으로 구동
echo "🚀 Elasticsearch 시작 중..."
./elasticsearch-8.8.0/bin/elasticsearch -d

# 5. Elasticsearch가 완전히 시작될 때까지 대기
echo "⏳ Elasticsearch 시작 대기 중 (60초)..."
sleep 60

# 6. Elasticsearch 비밀번호 재설정
echo "🔑 Elasticsearch 비밀번호 재설정 중..."
echo ""
echo "================================================"
echo "다음 질문에 'y'를 입력하고 Enter를 누르세요!"
echo "================================================"
echo ""
./elasticsearch-8.8.0/bin/elasticsearch-reset-password -u elastic

echo ""
echo "✅ 설치 완료!"
echo ""
echo "📝 다음 단계:"
echo "1. 위에서 출력된 비밀번호를 복사하세요"
echo "2. .env 파일을 열어 ELASTICSEARCH_PASSWORD에 붙여넣으세요"
echo "   nano .env"
echo ""
echo "🔍 Elasticsearch 실행 확인:"
echo "   curl -k -u elastic:YOUR_PASSWORD https://localhost:9200"
echo ""
