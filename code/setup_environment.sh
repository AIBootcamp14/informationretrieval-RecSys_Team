#!/bin/bash

# RAG 프로젝트 환경 설정 및 Baseline 실행 스크립트
#
# 이 스크립트는 다음 작업을 순서대로 수행합니다:
# 1. Elasticsearch 설치 및 설정 (Docker 사용)
# 2. Python 의존성 패키지 설치
# 3. Baseline 코드 실행 (문서 인덱싱 및 평가)

set -e # 스크립트 실행 중 오류 발생 시 즉시 중단

echo "=============== [Phase 1/3] Elasticsearch 설치 및 설정 시작 ==============="
# Elasticsearch 설치 스크립트 실행
chmod +x ./install_elasticsearch.sh
./install_elasticsearch.sh

# 생성된 비밀번호를 .env 파일에 자동으로 추가
# install_elasticsearch.sh가 ELASTIC_PASSWORD를 출력한다고 가정
PASSWORD=$(docker exec es01 /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic -b | grep -oP '(?<=New value: ).*')

if [ -f ".env" ] && grep -q "ELASTICSEARCH_PASSWORD" .env; then
    echo "ELASTICSEARCH_PASSWORD가 .env 파일에 이미 존재합니다. 값을 업데이트합니다."
    sed -i.bak "s/ELASTICSEARCH_PASSWORD=.*/ELASTICSEARCH_PASSWORD=${PASSWORD}/" .env && rm .env.bak
else
    echo "ELASTICSEARCH_PASSWORD를 .env 파일에 추가합니다."
    echo -e "\nELASTICSEARCH_PASSWORD=${PASSWORD}" >> .env
fi

echo "Elasticsearch 비밀번호가 .env 파일에 성공적으로 설정되었습니다."
echo "=============== [Phase 1/3] Elasticsearch 설정 완료 ==============="

echo "\n=============== [Phase 2/3] 의존성 패키지 설치 시작 ==============="
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "의존성 패키지 설치가 완료되었습니다."
else
    echo "오류: requirements.txt 파일을 찾을 수 없습니다."
    exit 1
fi
echo "=============== [Phase 2/3] 의존성 패키지 설치 완료 ==============="

echo "\n=============== [Phase 3/3] Baseline 코드 실행 테스트 시작 ==============="
python rag_with_elasticsearch.py
echo "=============== [Phase 3/3] Baseline 코드 실행 완료 ==============="

echo "\n🎉 모든 설정 및 Baseline 실행이 성공적으로 완료되었습니다."