# 실시간 정보 제공 프로젝트

***

## 11조 팀 구성
| 이름 | 팀 내 역할 | 분담된 역할 |
| ---------- | ---------- | ---------- |
| 인정배 | 팀장 | 모델 학습, 코드 작성, 데이터 전처리 및 RAG체인, Streamlit 오류해결 |
| 오승진 | 서기 | SA문서, REDEME작성, PPT 작성 |
| 박윤지 | 팀원 | 모델 학습, 코드 작성, 깃허브 관리, 지라 관리, streamlit 메이킹 |
| 이유림 | 팀원 | PPT 관리, PPT 작성, 시연 영상 작성 |

#### 공통된 역할 : API키 관리 (전원참여), 발표자 (전원참여)

## 프로젝트 소개
```
2024년 최근을 기준으로 실시간으로 어떤일이 발생하고 있는지
인공지능에게 물어봐서 뉴스를 한눈에 알 수 있는 인공지능 챗봇 입니다.
```
***

## 프로젝트 구조
[🔑프로젝트 구조](https://excalidraw.com/#json=QuYLlVgxDJso7_o-aGJHG,rBnzBwNOyYYGqi7i_7QJWw)
## 프로젝트 PPT
[✨프로젝트 보고서](https://docs.google.com/presentation/d/1Es9X6uiWgfBH_jLD8_vWzjJNfTw97hQ_kcZoeq4DXLA/edit#slide=id.p1)
***

## 주요 기능
```
2024년 뉴스의 실시간 정보를 불러오게 해서, 5개의 줄로 뉴스를 간단히 요약해줍니다.
최근뉴스에 어떤 기사가 올라 왔는지 텍스트로 한눈에 알아볼 수 있습니다.
```
***

## 구현 기술
```
사용자의 질문에 알맞은 응답대응
실시간 정보 업데이트
```
***

## 설치 라이브러리 
| 라이브러리 |
| ---------- |
| langchain_openai |
| langchain_core.messages |
| langchain.document_loaders |
| langchain.text_splitter |
| langchain.vectorstores.base |
| langchain_core.prompts |
| langchain_core.runnables |
| langchain.chains |
***

## 문제 해결 방안
```
대화 내용이 저장이 안되는 문제점이 발생함(메모리 기능 정상작동X). << 

OPEN_API_KEY를 대입하는 과정에서 환경변수 문제가 발생함.(최장범 매니저님을 통해 API키를 새로 발급받았으나, 새로 받은 키를 불러오지 않고 기존에 있던 키를 불러옴.) << 천준석 튜터님을 통해 해결 되었음.

```
***



***
## 결과

***

## 




