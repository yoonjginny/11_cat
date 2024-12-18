# 최근 뉴스 요약 챗봇

## 🌐 최근 뉴스
> * 최근 뉴스를 불러와서, 어떤 일이 발생했는지 알 수 있습니다.
> * (장르)뉴스를 입력하여 원하는 장르의 뉴스를 확인할 수 있습니다. - ex)정치,연애 등등...

## 11조 팀 구성
| 이름 | 팀 내 역할 | 분담된 역할 |
| ---------- | ---------- | ---------- |
| 인정배 | 팀장 | 모델 학습, 코드 작성, streamlit 코드 작성, Streamlit 오류해결 |
| 오승진 | 서기 | SA문서, REDEME작성, PPT 작성, 구성도 작성 |
| 박윤지 | 팀원 | 모델 학습, 코드 작성, 깃허브 관리, 지라 관리, streamlit 코드작성 및 인터페이스 |
| 이유림 | 팀원 | PPT 관리, PPT 작성, 구성도 작성, 시연영상 녹화 |

#### 공통된 역할 : API키 관리 (전원참여), 발표자 (전원참여)

## 프로젝트 소개
```
2024년 최근을 기준으로 실시간으로 어떤일이 발생하고 있는지
인공지능에게 물어봐서 뉴스를 한눈에 알 수 있는 인공지능 챗봇 입니다.
```
***
## 와이어프레임
[와이어프레임](https://excalidraw.com/#json=tI0fs8StXa9WkjrdJbVVD,lqCruYQVFbHg6S5Emod4cA)
## 아키텍처
[아키텍처](https://excalidraw.com/#json=67Et7apTC74w7oLYsD3v8,6vt8zScruhEXk60kO29j_g)
## 다이어그램
[다이어그램](https://excalidraw.com/#json=noJGN4bIiiTgS1YXiA-Md,YIPBKd4Dfe3XO93wdfEsig)
## 프로젝트 PPT
[✨프로젝트 보고서](https://docs.google.com/presentation/d/1Es9X6uiWgfBH_jLD8_vWzjJNfTw97hQ_kcZoeq4DXLA/edit#slide=id.p1)
***

## 주요 기능
```
2024년 최근 뉴스의 정보를 불러오게 해서, 5개의 줄로 뉴스를 간단히 요약해줍니다.
최근뉴스에 어떤 기사가 올라 왔는지 텍스트로 한눈에 알아볼 수 있습니다.
장르별 뉴스 검색이 가능합니다.
```
***

## 구현 기술
```
사용자의 질문에 알맞은 응답대응
실시간 정보 업데이트
대화 질문 시간대 특정 경로에서 저장하는 코드 
RAG체인
API 호출 외부 데이터 셋 받아내는 기술
문서 청크하는 기술 
vscode, jupyter notebook, github, 지라 , streamlit 기능구현  
```
***

## 문제 해결 방안
> * decoment 코드 형식이 맞지않아서 정확히는 context 즉 메타데이터와 여러가지 정보를 고르는 곳에서 변수 값이 호환이 안되어 변수가 생겼다. 전역의 변수 연계 코드를 찾아 변수명을 다르게 고쳐서 해결하였다.
> * 고유값이 필요했고 고유값이 없을 때 발생한 오류가 일어났다. 오류를 해결하기위해 키를 생성했고 그것만으론 부족해서 반복문에 i라는 변수를 주고 질문할 때 마다 1의 값을 주게하고 무한 반복하여 고유성을 주었다. 그럼에도 질문을 입력하지 않았으나 무한히 답을 해버리는 일이 발생했고 if문을 주어 막았으나 고유키가 고유하지 않아 오류가 발생해서 튜터님께 도움을 받았다.
[트러블 슈팅 사진](https://www.notion.so/teamsparta/1382dc3ef51481cdba3eef78bface315)
***

***
## 결과
[🎈프로젝트 결과물](https://www.notion.so/teamsparta/1382dc3ef5148165a486d8762ac90467)
***




