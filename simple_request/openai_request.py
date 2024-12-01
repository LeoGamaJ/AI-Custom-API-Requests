import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional, Union
from datetime import datetime
import json

class OpenAIChat:
    def __init__(self):
        # Tenta carregar as variáveis de ambiente
        load_dotenv()
        
        # Tenta obter a API key primeiro do ambiente, depois do .env
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        # Se não encontrar a API key, orienta o usuário
        if not self.api_key:
            print("API Key não encontrada. Por favor, siga os passos abaixo:")
            print("1. Crie um arquivo .env na raiz do projeto")
            print("2. Adicione sua API key no formato: OPENAI_API_KEY=sua_chave_aqui")
            print("3. Reinicie o script")
            raise ValueError("OPENAI_API_KEY não encontrada")
            
        # Inicializa o cliente OpenAI
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"Erro ao inicializar cliente OpenAI: {str(e)}")
            raise
            
        # Histórico de conversas
        self.conversation_history: List[Dict[str, str]] = []
        
        # Modelos disponíveis
        self.available_models = [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4',
            'gpt-4-turbo',
            'gpt-3.5-turbo',
            'o1-preview',
            'o1-mini'
        ]
        
        # Configurações padrão
        self.current_config = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'top_p': 1.0,
            'max_tokens': None,
            'presence_penalty': 0,
            'frequency_penalty': 0,
            'stream': True,
            'language': 'pt-br'
        }

        # Mensagens do sistema para cada idioma
        self.system_messages = {
            'pt-br': "Você é um assistente prestativo. Responda sempre em português do Brasil de forma clara e natural.",
            'en': "You are a helpful assistant. Always respond in English in a clear and natural way."
        }

    def create_chat_params(self, messages: List[Dict[str, str]]) -> Dict:
        """Cria os parâmetros para a chamada da API."""
        system_message = {
            "role": "system",
            "content": self.system_messages[self.current_config['language']]
        }
        
        full_messages = [system_message] + messages

        params = {
            "model": self.current_config['model'],
            "messages": full_messages,
            "temperature": self.current_config['temperature'],
            "top_p": self.current_config['top_p'],
            "presence_penalty": self.current_config['presence_penalty'],
            "frequency_penalty": self.current_config['frequency_penalty'],
            "stream": self.current_config['stream']
        }

        if self.current_config['max_tokens'] is not None:
            params["max_tokens"] = self.current_config['max_tokens']

        return params

    def send_message(self, message: str) -> Dict:
        """Envia uma mensagem para a API e retorna a resposta."""
        try:
            # Adiciona a mensagem do usuário ao histórico
            self.conversation_history.append({"role": "user", "content": message})
            
            # Cria a stream de chat
            stream = self.client.chat.completions.create(
                **self.create_chat_params(self.conversation_history)
            )
            
            response_content = ""
            
            # Processa a resposta
            if self.current_config['stream']:
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        response_content += content
                print()
            else:
                response_content = stream.choices[0].message.content
                
            # Adiciona a resposta ao histórico
            self.conversation_history.append({
                "role": "assistant",
                "content": response_content
            })

            return {
                'content': response_content,
                'citations': []
            }

        except Exception as e:
            error_msg = "Erro na requisição" if self.current_config['language'] == 'pt-br' else "Request error"
            print(f"{error_msg}: {str(e)}")
            return {'error': str(e)}

    def save_conversation(self) -> None:
        """Salva a conversa atual em um arquivo markdown."""
        try:
            # Cria o diretório 'historico' se não existir
            if not os.path.exists('historico'):
                os.makedirs('historico')
            
            # Gera nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historico/openai_chat_{timestamp}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Cabeçalho com informações da conversa
                f.write(f"# Conversa OpenAI - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
                
                # Configurações utilizadas
                f.write("## Configurações\n")
                for key, value in self.current_config.items():
                    f.write(f"- **{key}**: `{value}`\n")
                f.write("\n")
                
                # Conversa
                f.write("## Conversa\n\n")
                for msg in self.conversation_history:
                    role = "🧑 Usuário" if msg["role"] == "user" else "🤖 Assistente"
                    f.write(f"### {role}\n{msg['content']}\n\n")
            
            print(f"\n\033[92mConversa salva com sucesso em: {filename}\033[0m")
            
        except Exception as e:
            print(f"\n\033[91mErro ao salvar conversa: {str(e)}\033[0m")

    def clear_conversation(self) -> None:
        """Limpa o histórico da conversa."""
        self.conversation_history = []
        clear_msg = "Histórico de conversa limpo" if self.current_config['language'] == 'pt-br' else "Conversation history cleared"
        print(clear_msg)

    def show_current_config(self):
        """Mostra as configurações atuais."""
        title = "\nConfigurações atuais:" if self.current_config['language'] == 'pt-br' else "\nCurrent settings:"
        print(title)
        for key, value in self.current_config.items():
            print(f"{key}: {value}")

    def configure_settings(self):
        """Interface para configurar os parâmetros do chat."""
        try:
            is_ptbr = self.current_config['language'] == 'pt-br'
            
            print("\n=== " + ("Configuração do Chat" if is_ptbr else "Chat Configuration") + " ===")
            print("Pressione Enter para manter o valor atual" if is_ptbr else "Press Enter to keep current value")
            
            # Configuração de idioma
            print("\n" + ("Idiomas disponíveis:" if is_ptbr else "Available languages:"))
            print("1. Português (pt-br)\n2. English (en)")
            lang_choice = input("Escolha o idioma (1-2)" if is_ptbr else "Choose language (1-2): ")
            if lang_choice == '1':
                self.current_config['language'] = 'pt-br'
            elif lang_choice == '2':
                self.current_config['language'] = 'en'
            
            is_ptbr = self.current_config['language'] == 'pt-br'

            # Configuração do modelo
            print(f"\n" + ("Modelos disponíveis:" if is_ptbr else "Available models:"))
            for idx, model in enumerate(self.available_models, 1):
                print(f"{idx}. {model}")
            model_prompt = "Escolha o número do modelo" if is_ptbr else "Choose model number"
            model_choice = input(f"{model_prompt} (atual: {self.current_config['model']}): ")
            if model_choice.isdigit() and 1 <= int(model_choice) <= len(self.available_models):
                self.current_config['model'] = self.available_models[int(model_choice)-1]

            # Configuração de temperatura
            temp_prompt = "Temperature (0-2)" if is_ptbr else "Temperature (0-2)"
            temp = input(f"{temp_prompt} (atual: {self.current_config['temperature']}): ")
            if temp:
                self.current_config['temperature'] = float(temp)

            # Configuração de top_p
            top_p = input(f"Top P (0-1) (atual: {self.current_config['top_p']}): ")
            if top_p:
                self.current_config['top_p'] = float(top_p)

            # Configuração de max_tokens
            tokens_prompt = "Max Tokens" if is_ptbr else "Max Tokens"
            max_tokens = input(f"{tokens_prompt} (atual: {self.current_config['max_tokens']}): ")
            if max_tokens:
                self.current_config['max_tokens'] = int(max_tokens) if max_tokens.lower() != 'none' else None

            # Configuração de streaming
            stream_prompt = "Usar streaming (true/false)" if is_ptbr else "Use streaming (true/false)"
            stream = input(f"{stream_prompt} (atual: {self.current_config['stream']}): ")
            if stream.lower() in ['true', 'false']:
                self.current_config['stream'] = stream.lower() == 'true'

            print("\n" + ("Configurações atualizadas!" if is_ptbr else "Settings updated!"))
            self.show_current_config()
        except Exception as e:
            error_msg = "Erro ao configurar" if is_ptbr else "Configuration error"
            print(f"{error_msg}: {str(e)}")

def run_chat():
    """Função principal que executa o chat."""
    try:
        chat = OpenAIChat()
        
        print("\n" + "="*50)
        print("Bem-vindo ao Chat OpenAI!".center(50))
        print("="*50 + "\n")

        def print_help():
            """Função auxiliar para imprimir o menu de ajuda."""
            print("\nComandos disponíveis:")
            print("  1. Comandos básicos:")
            print("     - 'ajuda' ou '?': Mostra este menu")
            print("     - 'sair' ou 'q': Encerra o chat")
            print("     - 'limpar' ou 'cls': Limpa o histórico")
            print("     - 'salvar' ou 's': Salva conversa em arquivo")
            print("\n  2. Configurações:")
            print("     - 'config': Mostra configurações atuais")
            print("     - 'config [param]=[valor]': Altera configuração")
            print("     Exemplos:")
            print("     - config model=gpt-3.5-turbo")
            print("     - config temperature=0.8")
            print("\n  3. Modelos disponíveis:")
            for model in chat.available_models:
                print(f"     - {model}")
            print("\n  4. Parâmetros configuráveis:")
            print("     - model: Modelo a ser usado")
            print("     - temperature (0.0 a 1.0): Criatividade das respostas")
            print("     - top_p (0.0 a 1.0): Diversidade do texto")
            print("     - max_tokens: Limite de tokens na resposta")
            print("     - presence_penalty (-2.0 a 2.0): Penalidade por repetição")
            print("     - frequency_penalty (-2.0 a 2.0): Penalidade por frequência")
            print("     - language (pt-br/en): Idioma das respostas")

        def print_config():
            """Função auxiliar para imprimir configurações atuais."""
            print("\nConfigurações atuais:")
            for key, value in chat.current_config.items():
                print(f"  - {key}: {value}")

        # Mostrar ajuda inicial
        print_help()
        print("\nDigite sua mensagem ou comando. Use 'ajuda' para ver os comandos disponíveis.\n")
        
        while True:
            try:
                user_input = input("\n\033[94mVocê:\033[0m ").strip()
                
                if not user_input:
                    continue
                    
                # Comandos básicos
                if user_input.lower() in ['sair', 'q']:
                    print("\nEncerrando o chat...")
                    break
                elif user_input.lower() in ['limpar', 'cls']:
                    chat.conversation_history = []
                    print("\nHistórico limpo. Iniciando nova conversa.")
                    continue
                elif user_input.lower() in ['ajuda', '?']:
                    print_help()
                    continue
                elif user_input.lower() == 'config':
                    print_config()
                    continue
                elif user_input.lower().startswith('config '):
                    try:
                        config_str = user_input[7:]  # Remove 'config '
                        if not config_str:
                            print_config()
                            continue
                            
                        config_dict = dict(item.split('=') for item in config_str.split())
                        for key, value in config_dict.items():
                            if key in chat.current_config:
                                # Converter valores para o tipo apropriado
                                if key in ['temperature', 'top_p', 'presence_penalty', 'frequency_penalty']:
                                    value = float(value)
                                elif key == 'max_tokens':
                                    value = int(value) if value.lower() != 'none' else None
                                elif key == 'stream':
                                    value = value.lower() == 'true'
                                chat.current_config[key] = value
                            else:
                                print(f"\n\033[93mAviso: Configuração '{key}' desconhecida e será ignorada\033[0m")
                        print("\n\033[92mConfigurações atualizadas com sucesso!\033[0m")
                        print_config()
                    except Exception as e:
                        print(f"\n\033[91mErro ao atualizar configurações: {str(e)}\033[0m")
                        print("Use 'ajuda' para ver exemplos de uso do comando config.")
                    continue
                
                elif user_input.lower() in ['salvar', 's']:
                    chat.save_conversation()
                    continue
                
                # Enviar mensagem para o chat
                if chat.current_config['stream']:
                    print("\n\033[93mAssistente:\033[0m", end=" ")
                else:
                    print("\n\033[93mAssistente:\033[0m")
                result = chat.send_message(user_input)
                
                if "error" in result:
                    print("\n\033[91mOcorreu um erro. Use 'ajuda' para ver os comandos disponíveis.\033[0m")
                elif not chat.current_config['stream']:
                    print(result['content'])
                print()  # Linha extra para melhor legibilidade
                
            except KeyboardInterrupt:
                print("\n\nEncerrando o chat...")
                break
            except Exception as e:
                print(f"\n\033[91mErro: {str(e)}\033[0m")
                print("Use 'ajuda' para ver os comandos disponíveis.")
    
    except Exception as e:
        print(f"\n\033[91mErro fatal: {str(e)}\033[0m")
        print("O chat será encerrado.")

if __name__ == '__main__':
    run_chat()
