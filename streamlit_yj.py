import streamlit as st
import tiktoken
from loguru import logger

#from langchain.chains import ConversationalRetrievalChain

from langchain.memory import ConversationBufferMemory


from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory



import streamlit as st
import tiktoken
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory 

def main():
    st.set_page_config(
        page_title='NewsBot'
        page_icon=':robot:'
    )

    st.title('_News ChatBot : red[NewsBot]_ :robot:')

    #if 'conversation' not in st.session_state:
    #    st.session_state.chat_conversation = None

    #if 'chat_history' not in st.session_state:
    #    st.session_state.chat_history = None


    # 챗봇의 시작 
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 최신 뉴스를 물어봐주세요 :)"}]
    # 사용자 메시지 틀
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)



if __name__ == '__main__':
    main()

