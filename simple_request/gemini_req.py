import os
from dotenv import load_dotenv
import requests
import base64
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum
from PIL import Image
import io

class GeminiModel(Enum):
    GEMINI_PRO = "gemini-1.5-pro"
    GEMINI_FLASH = "gemini-1.5-flash"
    GEMINI_FLASH_8B = "gemini-1.5-flash-8b"

class GeminiChat:
    def __init__(self):
        # Carregar variáveis de ambiente
        load_dotenv()
        
        # Obter a API key
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            print("API Key não encontrada. Por favor, siga os passos abaixo:")
            print("1. Crie um arquivo .env na raiz do projeto")
            print("2. Adicione sua API key no formato: GEMINI_API_KEY=sua_chave_aqui")
            print("3. Reinicie o script")
            raise ValueError("GEMINI_API_KEY não encontrada")
        
        # Histórico de conversas
        self.conversation_history: List[Dict[str, str]] = []
        
        # Modelos disponíveis
        self.available_models = [model.value for model in GeminiModel]
        
        # Configurações padrão
        self.current_config = {
            'model': GeminiModel.GEMINI_PRO.value,
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.95,
            'max_tokens': 2048,
            'language': 'pt-br'
        }

        # Mensagens do sistema para cada idioma
        self.system_messages = {
            'pt-br': "Você é um assistente prestativo. Responda sempre em português do Brasil de forma clara e natural.",
            'en': "You are a helpful assistant. Always respond in English in a clear and natural way."
        }

    def update_config(self, **kwargs) -> None:
        """Atualiza as configurações do modelo."""
        valid_keys = self.current_config.keys()
        for key, value in kwargs.items():
            if key in valid_keys:
                if key == 'model' and value not in self.available_models:
                    raise ValueError(f"Modelo '{value}' não disponível. Escolha entre: {', '.join(self.available_models)}")
                elif key == 'temperature' and not (0 <= float(value) <= 1):
                    raise ValueError("Temperature deve estar entre 0 e 1")
                self.current_config[key] = value
            else:
                print(f"Aviso: Configuração '{key}' desconhecida e será ignorada")

    def process_image(self, image_path: str) -> Dict:
        """Processa uma imagem para envio à API."""
        try:
            with Image.open(image_path) as img:
                # Converter para RGB se necessário
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionar se a imagem for muito grande
                max_size = 2048
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Converter para bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                img_bytes = img_byte_arr.getvalue()

                return {
                    "mimeType": "image/jpeg",
                    "data": base64.b64encode(img_bytes).decode('utf-8')
                }
        except Exception as e:
            raise ValueError(f"Erro ao processar imagem: {str(e)}")

    def create_request_data(self, content: Union[str, Dict], is_image: bool = False) -> Dict:
        """Cria o payload para a requisição."""
        if is_image:
            parts = [
                {"text": content["prompt"]},
                {
                    "inline_data": {
                        "mimeType": content["image_data"]["mimeType"],
                        "data": content["image_data"]["data"]
                    }
                }
            ]
        else:
            parts = [{"text": str(content)}]

        return {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "temperature": self.current_config['temperature'],
                "topK": self.current_config['top_k'],
                "topP": self.current_config['top_p'],
                "maxOutputTokens": self.current_config['max_tokens']
            }
        }

    def send_message(self, message: str, image_path: Optional[str] = None) -> Dict:
        """Envia uma mensagem para a API e retorna a resposta."""
        try:
            # Processa imagem se fornecida
            if image_path:
                self.current_config['model'] = GeminiModel.GEMINI_FLASH_8B.value
                image_data = self.process_image(image_path)
                content = {
                    "prompt": message or "Descreva esta imagem em detalhes",
                    "image_data": image_data
                }
                is_image = True
            else:
                content = f"{self.system_messages[self.current_config['language']]}\n\n{message}"
                is_image = False

            # Adiciona a mensagem do usuário ao histórico
            self.conversation_history.append({"role": "user", "content": message})
            
            # Prepara a URL e os dados
            url = f"https://generativelanguage.googleapis.com/v1/models/{self.current_config['model']}:generateContent?key={self.api_key}"
            data = self.create_request_data(content, is_image)
            
            # Faz a requisição
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            # Processa a resposta
            response_data = response.json()
            if 'candidates' in response_data and response_data['candidates']:
                content = response_data['candidates'][0]['content']
                if 'parts' in content and content['parts']:
                    response_text = content['parts'][0].get('text', '')
                    print(response_text)
                    
                    # Adiciona a resposta ao histórico
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    return {
                        "response": response_text,
                        "conversation_history": self.conversation_history
                    }
            
            return {"error": "Não foi possível gerar uma resposta."}
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Erro na requisição: {str(e)}"
            if hasattr(e.response, 'json'):
                try:
                    error_data = e.response.json()
                    if 'error' in error_data:
                        error_msg = f"Erro: {error_data['error'].get('message', str(e))}"
                except:
                    pass
            print(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Erro inesperado: {str(e)}"
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
            filename = f"historico/gemini_chat_{timestamp}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Cabeçalho com informações da conversa
                f.write(f"# Conversa Gemini - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
                
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
            
            print(f"\nConversa salva com sucesso em: {filename}")
            
        except Exception as e:
            print(f"Erro ao salvar conversa: {str(e)}")

def run_chat():
    """Função principal que executa o chat."""
    try:
        chat = GeminiChat()
        
        print("\n" + "="*50)
        print("Bem-vindo ao Chat Gemini!".center(50))
        print("="*50 + "\n")
        print("Comandos disponíveis:")
        print("- 'sair' ou 'q': Encerra o chat")
        print("- 'limpar' ou 'cls': Limpa o histórico")
        print("- 'salvar' ou 's': Salva a conversa")
        print("- 'config modelo=nome': Altera o modelo")
        print("- 'imagem caminho': Analisa uma imagem")
        print(f"\nModelos disponíveis: {', '.join(chat.available_models)}\n")
        
        while True:
            try:
                user_input = input("\nVocê: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['sair', 'q']:
                    print("\nEncerrando o chat...")
                    break
                elif user_input.lower() in ['limpar', 'cls']:
                    chat.clear_history()
                    continue
                elif user_input.lower() in ['salvar', 's']:
                    chat.save_conversation()
                    continue
                elif user_input.lower().startswith('config '):
                    try:
                        config_str = user_input[7:]
                        config_dict = dict(item.split('=') for item in config_str.split())
                        chat.update_config(**config_dict)
                        print("\nConfigurações atualizadas!")
                    except Exception as e:
                        print(f"\nErro ao atualizar configurações: {str(e)}")
                    continue
                elif user_input.lower().startswith('imagem '):
                    image_path = user_input[7:].strip()
                    if not os.path.exists(image_path):
                        print(f"\nErro: Imagem não encontrada: {image_path}")
                        continue
                    prompt = input("\nDigite uma descrição ou pergunta sobre a imagem: ")
                    chat.send_message(prompt, image_path)
                    continue
                
                print("\nAssistente:", end=" ")
                chat.send_message(user_input)
                
            except KeyboardInterrupt:
                print("\n\nOperação cancelada pelo usuário.")
                continue
            except Exception as e:
                print(f"\nErro: {str(e)}")
                continue
    
    except Exception as e:
        print(f"\nErro fatal: {str(e)}")

if __name__ == '__main__':
    run_chat()
