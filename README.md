# Detecção de Veículos e EPIs (Rede Neural)

Este projeto utiliza redes neurais para realizar a detecção de veículos, pessoas e capacetes (EPIs). O código foi desenvolvido utilizando o modelo **YOLOv8**, e é capaz de:

- Detectar veículos em movimento.
- Identificar pessoas em um vídeo.
- Verificar se as pessoas estão utilizando capacetes.
- Salvar imagens de **ROIs** (Regiões de Interesse) de pessoas sem capacete.
- Gerar um log de alertas sempre que uma pessoa sem capacete for detectada.

## Alteração Importante: Vídeo de Entrada

Devido ao tamanho do arquivo `ch5.mp4`, ele foi removido do repositório. O link para o vídeo está disponível no arquivo `data/input/link_do_video.txt`. Para executar o código, insira o caminho do vídeo no script principal (`main.py`) antes de iniciar o processamento.

## Como Funciona o Código

O código realiza as seguintes etapas:

1. **Carregamento do modelo YOLOv8**: O modelo treinado é carregado a partir do arquivo `best.pt`, utilizando a biblioteca `ultralytics`.

2. **Processamento de vídeo**: O vídeo de entrada é processado frame por frame. Para cada frame, o modelo YOLOv8 realiza a detecção de objetos, incluindo veículos, pessoas e capacetes.

3. **Verificação de capacetes**: A detecção de pessoas é comparada com a detecção de capacetes. Se uma pessoa estiver detectada sem capacete, sua imagem (ROI) é salva e um alerta é gerado.

4. **Geração de resultados**: O código salva as imagens de ROIs para as pessoas sem capacete, além de salvar imagens de veículos. O log de alertas é gerado para cada pessoa sem capacete detectada.

5. **Saída de vídeo**: O vídeo de entrada é processado e gerado um vídeo de saída contendo as deteções e informações sobre os objetos identificados, com boxes desenhados ao redor de veículos e pessoas.

## Funcionalidades

- **Detecção de Veículos**: O código detecta veículos em movimento e salva as imagens (ROIs) de cada veículo identificado.
- **Verificação de Capacetes (EPI)**: O sistema verifica se as pessoas detectadas estão utilizando capacete. Se não, o código salva a imagem da pessoa sem capacete e gera um alerta.
- **Armazenamento de ROIs**: As regiões de interesse (ROIs) das pessoas e veículos detectados são salvas em pastas específicas.
- **Log de Alertas**: Sempre que uma pessoa sem capacete é detectada, um log com a descrição e localização da ROI é registrado.

## Resultados

Após a execução do código, as seguintes saídas são geradas:

- **Vídeo de saída**: O vídeo processado (`output_video.mp4`) é salvo na pasta `output`, contendo todas as deteções realizadas pelo modelo.
- **Imagens de ROIs**: As imagens das pessoas sem capacete e veículos detectados são salvas nas pastas `roi_pessoas` e `roi_carros`, respectivamente.
- **Logs de alertas**: Um arquivo de log (`alertas.log`) é gerado com os alertas sempre que uma pessoa sem capacete for detectada.

Embora o código consiga salvar corretamente as imagens de ROIs de pessoas sem capacete, há um pequeno problema em que, em algumas ocasiões, imagens de pessoas com capacete são salvas. Este é um ponto de melhoria a ser trabalhado, buscando aumentar a precisão da detecção de capacetes, especialmente em casos em que a pessoa está em movimento.

## Estrutura de Diretórios

```plaintext
.
├── data
│   ├── input
│   │   ├── link_do_video.txt  # Link do vídeo de entrada (ch5.mp4 no YouTube)
│   └── output
│       ├── roi_carros         # Pastas com as ROIs dos veículos
│       ├── roi_pessoas        # Pastas com as ROIs das pessoas
│       └── output_video.mp4   # Vídeo de saída
├── logs
│   └── alertas.log            # Arquivo de log contendo os alertas
├── models
│   └── best.pt                # Modelo treinado YOLO
├── src
│   └── main.py                # Código principal para executar o projeto
├── .gitignore
├── README.md                  # Este arquivo
├── requirements.txt           # Arquivo com as dependências do projeto
```

## Como Executar

### Requisitos
Certifique-se de ter o **Python 3.7+** instalado. Além disso, o código depende das bibliotecas listadas no arquivo `requirements.txt`.

### Passos

1. **Clone o repositório:**
```
   git clone https://github.com/Davi-Tunussi/Detec-o_de_Ve-culos_e_EPIs.git
```

   cd Detec-o_de_Ve-culos_e_EPIs
### Instruções de Uso

2. **Instale as Dependências**

```
pip install -r requirements.txt
```

3. **Configure o Vídeo de Entrada**

   Acesse o arquivo data/input/link_do_video.txt e copie o link do vídeo no YouTube.
Atualize o script main.py com o caminho do vídeo de entrada (após baixá-lo ou usar streaming, conforme necessário).

4. **Execute o Script Principal**
   
      Para executar o código principal, rode o seguinte comando:
```
python src/main.py
```
   O código irá processar o vídeo de entrada e gerar o vídeo de saída com as detecções, além de salvar as imagens das ROIs e gerar os logs de alertas.

## Dependências
As principais dependências do projeto estão listadas no arquivo requirements.txt:

-opencv-python

-torch

-ultralytics

Para instalar as dependências automaticamente, execute:
```
pip install -r requirements.txt
```
### Possíveis Melhorias
- Aprimorar a Detecção de Capacetes
   O modelo pode ser ajustado para melhorar a detecção de capacetes, especialmente em pessoas em movimento, para evitar o erro de salvar ROIs de pessoas com capacete.

- Ajuste de Parâmetros de IOU
   O valor do Intersection over Union (IoU) utilizado para verificar se dois objetos são o mesmo pode ser ajustado para melhorar a precisão na comparação de objetos em movimento.

- Análise de Desempenho
   Melhorar o desempenho em termos de velocidade, especialmente em vídeos grandes, considerando a otimização do uso da GPU.

### Contribuindo
Sinta-se à vontade para contribuir com melhorias! Se você quiser fazer uma mudança, siga estas etapas:

- Faça um fork deste repositório.

- Crie uma branch para sua modificação:
```
git checkout -b minha-modificacao
```
- Faça suas mudanças e faça um commit:
```
git commit -am 'Adiciona nova funcionalidade'
```
- Envie para o repositório remoto:
```
git push origin minha-modificacao
```
- Abra um Pull Request para a branch principal.

## Documentação criada por Davi Tunussi.


