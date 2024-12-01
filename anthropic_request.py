import os
from dotenv import load_dotenv
import requests
from typing import List, Dict, Optional, Union
from datetime import datetime
import json

class AnthropicChat:
    def __init__(self):
        # Carregar variáveis de ambiente
        load_dotenv()
        
        # Obter a API key
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            print("API Key não encontrada. Por favor, siga os passos abaixo:")
            print("1. Crie um arquivo .env na raiz do projeto")
            print("2. Adicione sua API key no formato: ANTHROPIC_API_KEY=sua_chave_aqui")
            print("3. Reinicie o script")
            raise ValueError("ANTHROPIC_API_KEY não encontrada")
            
        # URL da API
        self.api_url = "https://api.anthropic.com/v1/messages"
        
        # Histórico de conversas
        self.conversation_history: List[Dict[str, str]] = []
        
        # Modelos disponíveis organizados por categoria
        self.model_categories = {
            'Claude 3.5 Sonnet': [
                'claude-3-sonnet-20240229',     # Versão específica
                'claude-3-sonnet'               # Última versão
            ],
            'Claude 3.5 Haiku': [
                'claude-3-haiku-20240307',      # Versão específica
                'claude-3-haiku'                # Última versão
            ]
        }
        
        # Lista completa de modelos
        self.available_models = [
            model
            for category in self.model_categories.values()
            for model in category
        ]
        
        # Configurações padrão
        self.current_config = {
            'model': 'claude-3-sonnet',  # Modelo padrão
            'temperature': 0.7,
            'max_tokens': 1000,
            'system': "Você é um prestativo assistente. Assuma o papel de especialista na pergunta realizada e responda.",
            'language': 'pt-br'
        }

        # Mensagens do sistema para cada idioma
        self.system_messages = {
            'pt-br': "Você é um prestativo assistente. Assuma o papel de especialista na pergunta realizada e responda.",
            'en': "You are a helpful assistant. Take on the role of an expert in the question asked and respond."
        }

    def update_config(self, **kwargs) -> None:
        """Atualiza as configurações do modelo."""
        valid_keys = self.current_config.keys()
        for key, value in kwargs.items():
            if key in valid_keys:
                if key == 'model' and value not in self.available_models:
                    raise ValueError(f"Modelo '{value}' não disponível. Escolha entre: {', '.join(self.available_models)}")
                if key == 'temperature' and not (0 <= float(value) <= 1):
                    raise ValueError("Temperature deve estar entre 0 e 1")
                if key == 'max_tokens' and int(value) <= 0:
                    raise ValueError("Max tokens deve ser maior que 0")
                self.current_config[key] = value
            else:
                print(f"Aviso: Configuração '{key}' desconhecida e será ignorada")

    def create_chat_params(self, messages: List[Dict[str, str]]) -> Dict:
        """Cria os parâmetros para a chamada da API."""
        return {
            "model": self.current_config['model'],
            "messages": messages,
            "temperature": self.current_config['temperature'],
            "max_tokens": self.current_config['max_tokens'],
            "system": self.current_config['system']
        }

    def send_message(self, message: str) -> Dict:
        """Envia uma mensagem para a API e retorna a resposta."""
        try:
            # Adiciona a mensagem do usuário ao histórico
            self.conversation_history.append({"role": "user", "content": message})
            
            # Prepara o cabeçalho da requisição
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Tenta fazer a requisição
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=self.create_chat_params(self.conversation_history)
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as api_error:
                error_msg = f"\nErro na API Anthropic: {str(api_error)}\n"
                if response.status_code == 404:
                    error_msg += f"Modelo '{self.current_config['model']}' não encontrado.\n"
                    error_msg += f"Modelos disponíveis: {', '.join(self.available_models)}\n"
                    error_msg += "Use o comando 'config model=nome-do-modelo' para alterar o modelo."
                print(error_msg)
                return {"error": error_msg}
            
            # Processa a resposta
            result = response.json()
            response_content = result['content'][0]['text']
            print(response_content)
            
            # Adiciona a resposta ao histórico
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            return {
                "response": response_content,
                "conversation_history": self.conversation_history
            }
            
        except Exception as e:
            error_msg = f"\nErro inesperado: {str(e)}\n"
            print(error_msg)
            return {"error": error_msg}

    def clear_history(self) -> None:
        """Limpa o histórico de conversas."""
        self.conversation_history = []
        print("Histórico de conversas limpo.")

    def save_conversation(self) -> None:
        """Salva a conversa atual em um arquivo markdown."""
        try:
            # Cria o diretório 'historico' se não existir
            if not os.path.exists('historico'):
                os.makedirs('historico')
            
            # Gera nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historico/anthropic_chat_{timestamp}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Cabeçalho com informações da conversa
                f.write(f"# Conversa Anthropic - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
                
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

def run_chat():
    """Função principal que executa o chat."""
    try:
        chat = AnthropicChat()
        
        print("\n" + "="*50)
        print("Bem-vindo ao Chat Anthropic!".center(50))
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
            print("     - config model=claude-3-sonnet")
            print("     - config temperature=0.8")
            print("     - config system='Seja um especialista em Python'")
            print("\n  3. Modelos disponíveis por categoria:")
            for category, models in chat.model_categories.items():
                print(f"\n     {category}:")
                for model in models:
                    print(f"     - {model}")
            print("\n  4. Parâmetros configuráveis:")
            print("     - model: Modelo a ser usado")
            print("     - temperature (0.0 a 1.0): Criatividade das respostas")
            print("     - max_tokens: Limite de tokens na resposta")
            print("     - system: Mensagem do sistema/instruções")
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
                    chat.clear_history()
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
                        chat.update_config(**config_dict)
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
