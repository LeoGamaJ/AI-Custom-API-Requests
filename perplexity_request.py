import os
from dotenv import load_dotenv
import requests
import json
from typing import List, Dict, Optional, Union
from datetime import datetime

class PerplexityChat:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY não encontrada no arquivo .env")
            
        self.url = 'https://api.perplexity.ai/chat/completions'
        self.conversation_history: List[Dict[str, str]] = []
        
        self.available_models = [
            'llama-3.1-sonar-small-128k-chat',
            'llama-3.1-sonar-large-128k-chat',
            'llama-3.1-sonar-small-128k-online',
            'llama-3.1-sonar-large-128k-online',
            'llama-3.1-sonar-huge-128k-online'
        ]
        
        # Configurações padrão
        self.current_config = {
            'model': 'llama-3.1-sonar-small-128k-online',
            'temperature': 0.2,
            'top_p': 0.9,
            'top_k': 0,
            'max_tokens': None,
            'presence_penalty': 0,
            'frequency_penalty': 1,
            'return_citations': True,
            'return_related_questions': False,
            'search_recency_filter': None,
            'language': 'pt-br'  # Adicionado configuração de idioma
        }

        # Mensagens do sistema para cada idioma
        self.system_messages = {
            'pt-br': "Você é um assistente prestativo. Responda sempre em português do Brasil de forma clara e natural.",
            'en': "You are a helpful assistant. Always respond in English in a clear and natural way."
        }

    def create_headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def create_request_body(self, messages: List[Dict[str, str]]) -> Dict:
        # Adiciona a mensagem do sistema no início da conversa
        system_message = {
            "role": "system",
            "content": self.system_messages[self.current_config['language']]
        }
        
        full_messages = [system_message] + messages

        body = {
            "model": self.current_config['model'],
            "messages": full_messages,
            "temperature": self.current_config['temperature'],
            "top_p": self.current_config['top_p'],
            "top_k": self.current_config['top_k'],
            "presence_penalty": self.current_config['presence_penalty'],
            "frequency_penalty": self.current_config['frequency_penalty'],
            "return_citations": self.current_config['return_citations'],
            "return_related_questions": self.current_config['return_related_questions']
        }

        if self.current_config['max_tokens'] is not None:
            body["max_tokens"] = self.current_config['max_tokens']
            
        if self.current_config['search_recency_filter']:
            body["search_recency_filter"] = self.current_config['search_recency_filter']

        return body

    def send_message(self, message: str) -> Dict:
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            response = requests.post(
                self.url,
                headers=self.create_headers(),
                data=json.dumps(self.create_request_body(self.conversation_history))
            )
            
            response.raise_for_status()
            result = response.json()
            
            assistant_message = result['choices'][0]['message']
            self.conversation_history.append(assistant_message)

            # Imprime a resposta do assistente
            print(assistant_message['content'])
            
            # Retorna o resultado completo
            return {
                'content': assistant_message['content'],
                'citations': assistant_message.get('citations', [])
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = "\nErro na requisição"
            if hasattr(e, 'response') and e.response is not None:
                print(f"{error_msg}: {e.response.text}")
            return {'error': str(e)}

    def save_conversation(self) -> None:
        """Salva a conversa atual em um arquivo markdown."""
        try:
            # Cria o diretório 'historico' se não existir
            if not os.path.exists('historico'):
                os.makedirs('historico')
            
            # Gera nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historico/perplexity_chat_{timestamp}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Cabeçalho com informações da conversa
                f.write(f"# Conversa Perplexity - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
                
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
                    
                    # Adiciona citações se houver
                    if msg.get("citations"):
                        f.write("#### 📚 Citações\n")
                        for citation in msg["citations"]:
                            f.write(f"- {citation}\n")
                        f.write("\n")
            
            print(f"\n\033[92mConversa salva com sucesso em: {filename}\033[0m")
            
        except Exception as e:
            print(f"\n\033[91mErro ao salvar conversa: {str(e)}\033[0m")

    def clear_conversation(self) -> None:
        self.conversation_history = []
        clear_msg = "Histórico de conversa limpo" if self.current_config['language'] == 'pt-br' else "Conversation history cleared"
        print(clear_msg)

    def show_current_config(self):
        title = "\nConfigurações atuais:" if self.current_config['language'] == 'pt-br' else "\nCurrent settings:"
        print(title)
        for key, value in self.current_config.items():
            print(f"{key}: {value}")

    def configure_settings(self):
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
        
        # Atualiza a variável is_ptbr após a possível mudança de idioma
        is_ptbr = self.current_config['language'] == 'pt-br'

        # Modelo
        print(f"\n" + ("Modelos disponíveis:" if is_ptbr else "Available models:"))
        for idx, model in enumerate(self.available_models, 1):
            print(f"{idx}. {model}")
        model_prompt = "Escolha o número do modelo" if is_ptbr else "Choose model number"
        model_choice = input(f"{model_prompt} (atual: {self.current_config['model']}): ")
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(self.available_models):
            self.current_config['model'] = self.available_models[int(model_choice)-1]

        # Temperature
        temp_prompt = "Temperature (0-2)" if is_ptbr else "Temperature (0-2)"
        temp = input(f"{temp_prompt} (atual: {self.current_config['temperature']}): ")
        if temp:
            self.current_config['temperature'] = float(temp)

        # Top P
        top_p = input(f"Top P (0-1) (atual: {self.current_config['top_p']}): ")
        if top_p:
            self.current_config['top_p'] = float(top_p)

        # Max Tokens
        tokens_prompt = "Max Tokens" if is_ptbr else "Max Tokens"
        max_tokens = input(f"{tokens_prompt} (atual: {self.current_config['max_tokens']}): ")
        if max_tokens:
            self.current_config['max_tokens'] = int(max_tokens) if max_tokens.lower() != 'none' else None

        # Search Recency
        print("\n" + ("Opções de recência de busca:" if is_ptbr else "Search recency options:"))
        print("1. month\n2. week\n3. day\n4. hour\n5. none")
        recency_prompt = "Escolha a recência (1-5)" if is_ptbr else "Choose recency (1-5)"
        recency = input(f"{recency_prompt} (atual: {self.current_config['search_recency_filter']}): ")
        if recency:
            recency_options = {
                '1': 'month',
                '2': 'week',
                '3': 'day',
                '4': 'hour',
                '5': None
            }
            self.current_config['search_recency_filter'] = recency_options.get(recency)

        # Citations
        citations_prompt = "Retornar citações (true/false)" if is_ptbr else "Return citations (true/false)"
        citations = input(f"{citations_prompt} (atual: {self.current_config['return_citations']}): ")
        if citations.lower() in ['true', 'false']:
            self.current_config['return_citations'] = citations.lower() == 'true'

        print("\n" + ("Configurações atualizadas!" if is_ptbr else "Settings updated!"))
        self.show_current_config()

def main():
    """Função principal que executa o chat."""
    try:
        chat = PerplexityChat()
        
        print("\n" + "="*50)
        print("Bem-vindo ao Chat Perplexity!".center(50))
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
            print("     - config model=llama-3.1-sonar-small-128k-online")
            print("     - config temperature=0.8")
            print("\n  3. Modelos disponíveis:")
            for model in chat.available_models:
                print(f"     - {model}")
            print("\n  4. Parâmetros configuráveis:")
            print("     - model: Modelo a ser usado")
            print("     - temperature (0.0 a 1.0): Criatividade das respostas")
            print("     - top_p (0.0 a 1.0): Diversidade do texto")
            print("     - top_k (0+): Número de tokens a considerar")
            print("     - max_tokens: Limite de tokens na resposta")
            print("     - presence_penalty (-2.0 a 2.0): Penalidade por repetição")
            print("     - frequency_penalty (-2.0 a 2.0): Penalidade por frequência")
            print("     - return_citations (true/false): Retornar citações")
            print("     - return_related_questions (true/false): Retornar perguntas relacionadas")
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
                                elif key in ['max_tokens', 'top_k']:
                                    value = int(value) if value.lower() != 'none' else None
                                elif key in ['return_citations', 'return_related_questions']:
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
                print("\n\033[93mAssistente:\033[0m", end=" ")
                result = chat.send_message(user_input)
                
                if "error" in result:
                    print("\n\033[91mOcorreu um erro. Use 'ajuda' para ver os comandos disponíveis.\033[0m")
                elif result.get("citations"):
                    print("\n\n\033[96mCitações:\033[0m")
                    for citation in result["citations"]:
                        print(f"  {citation}")
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

if __name__ == "__main__":
    main()