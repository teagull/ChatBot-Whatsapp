import os
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

os.environ['GROQ_API_KEY'] = config('GROQ_API_KEY')

class AIBot:
    def __init__(self):
        self.__chat = ChatGroq(model='gemma2-9b-it')
        self.__retriever = self.__build_retriever()

    def __build_retriever(self):
        persist_directory = '/app/chroma_data'
        embedding = HuggingFaceEmbeddings()
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        return vector_store.as_retriever(
            search_kwargs={'k': 30},
        )

    def __build_messages(self, history_messages, question):
        messages = []
        for message in history_messages:
            message_class = HumanMessage if message.get('fromMe') else AIMessage
            messages.append(message_class(content=message.get('body')))
        messages.append(HumanMessage(content=question))
        return messages

    def invoke(self, history_messages, question):
        SYSTEM_TEMPLATE = '''
        Você é o Professor Newton, um assistente especializado em Física Mecânica, criado para responder dúvidas de alunos de forma clara, 
        objetiva e humanizada. Seu objetivo é fornecer explicações diretas, respeitosas e acessíveis, como se estivesse em um bate-papo 
        natural, sempre em portugues brasileiro.
        Regras para Respostas:
        Foque exclusivamente em Física Mecânica.
        Se o usuário perguntar sobre outro assunto, responda educadamente que você foi treinado apenas para Física Mecânica.
        Mantenha um tom natural e amigável.
        Escreva de forma clara, como um professor conversando com um aluno.
        Use exemplos do dia a dia para tornar o conteúdo mais compreensível.
        Seja direto e didático.
        Explique conceitos de forma objetiva, sem respostas muito longas ou complexas.
        Se necessário, divida explicações em passos para facilitar o entendimento.
        Leve em consideração o histórico da conversa.
        Se um aluno já fez perguntas anteriores, conecte as respostas para manter a coerência.
        Demonstre paciência e incentivo ao aprendizado.
        Caso o aluno demonstre dificuldade, reforce a explicação com exemplos diferentes.
        <context>
        {context}
        </context>
        '''
        docs = self.__retriever.invoke(question)
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', SYSTEM_TEMPLATE),
                MessagesPlaceholder(variable_name='messages'),
            ]
        )
        document_chain = create_stuff_documents_chain(self.__chat, question_answering_prompt)
        response = document_chain.invoke(
            {
                'context': docs,
                'messages': self.__build_messages(history_messages, question),
            }
        )
        return response
