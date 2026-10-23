# 🔎 SkuScope

**Reconhecimento de SKUs em fotos e vídeos de gôndola.** O YOLO encontra cada produto na prateleira, e o DINO diz qual SKU cada um é. Num vídeo, cada produto é rastreado entre os quadros e contado uma vez só.

🇺🇸 [Read in English](README.md)

> Projeto de estudo, feito para aprendizado e experimentação. Não é para uso em produção.

![Resultado do SkuScope: cada garrafa de uma gôndola de refrigerantes marcada com seu SKU e a similaridade](docs/example-result.jpg)
<sub>Catálogo de 7 SKUs, **uma foto de cada**, sem treino. Caixas cinza são "unknown" (desconhecido). Foto: [Soft drink shelf](https://commons.wikimedia.org/wiki/File:Soft_drink_shelf.JPG), domínio público.</sub>

## Por que essa abordagem

O jeito clássico de reconhecer produtos é treinar um classificador com uma classe por SKU. A cada mudança de sortimento você precisa retreinar, e o modelo tende a "esquecer" os produtos antigos.

O SkuScope divide o problema em duas partes:

1. **Onde estão os produtos?** Um detector YOLO11 treinado no [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) encontra todos os produtos da prateleira. Ele só conhece uma classe: "produto".
2. **Qual produto é este?** Cada recorte vira um vetor (uma "impressão digital visual") com o [DINOv2](https://github.com/facebookresearch/dinov2). Esse vetor é comparado, por similaridade de cosseno, com os vetores das fotos do seu catálogo.

**Cadastrar um SKU novo é só colocar algumas fotos numa pasta.** Não há treino.

```
Foto da gôndola
   ↓  YOLO11 (SKU-110K)        encontra cada produto
Recortes dos produtos
   ↓  DINOv2                   um vetor por recorte
Vetores  ──►  similaridade de cosseno com cada foto do catálogo
   ↓
Similaridade com cada SKU (a da sua melhor foto)
   ↓
≥ limite e à frente do segundo colocado pela margem → SKU
dúvida só entre variantes de uma linha de produto   → a linha de produto
qualquer outro caso                                 → "unknown" (desconhecido)
```

### Por que DINO e não CLIP?

O CLIP aprendeu com legendas de imagens, então conhece *conceitos* ("uma lata de refrigerante de cola"). O DINO aprendeu só com imagens (auto-supervisão), então presta atenção em *detalhes visuais*: textura, logotipos, a faixa fina que separa o "Original" do "Zero". São esses detalhes que diferenciam SKUs parecidos.

## Resultados na gôndola de exemplo

Sete marcas entraram no catálogo com **uma foto recortada de cada**. Duas marcas (Traditional's Lemonade e Cream Soda, 10 garrafas ao todo) ficaram de fora de propósito, para ver como o modelo lida com produtos que não conhece. Cada uma das 48 detecções foi conferida manualmente.

| O quê | Resultado |
|---|---|
| Garrafas detectadas pelo YOLO | 48 (44 garrafas + 4 caixas parciais ou que não são produto, nas bordas) |
| **Marcas do catálogo, SKU mais parecido** | **34 de 34 corretas**, inclusive uma garrafa vista de costas |
| Similaridade dos acertos | quase todos entre 0,80 e 0,96 (o menor foi 0,66, a garrafa de costas) |
| Similaridade das marcas *fora* do catálogo | 0,67 a 0,84 |
| Similaridade das caixas parciais nas bordas | até 0,45 |
| Execução completa num MacBook M2 Pro | cerca de 7 s, incluindo o carregamento dos modelos (a detecção leva cerca de 1 s) |

**Conclusão:** escolher o SKU certo entre os conhecidos é a parte fácil. **O difícil é dizer "não conheço este produto".** Produtos desconhecidos ainda chegam a 0,84 de similaridade, então nenhum limite único separa tudo perfeitamente:

| Regra | Garrafas conhecidas com nome (de 34) | Garrafas desconhecidas com nome errado |
|---|---|---|
| Só o limite de 0,80 | 31 | 5 |
| Só o limite de 0,85 | 26 | 0 |
| **Limite de 0,80 + margem de 0,03** (o padrão) | **30** | **1** |

O que ajuda é a **margem**: um produto só recebe nome quando o SKU mais parecido supera o segundo colocado por pelo menos 0,03. Uma garrafa que o catálogo não conhece costuma se parecer um pouco com vários SKUs ao mesmo tempo, então os dois maiores scores ficam próximos; uma garrafa conhecida costuma ganhar com folga. A imagem acima usa o padrão.

Esses números são otimistas: os recortes do catálogo vieram da mesma foto, com a mesma luz e o mesmo ângulo. Com fotos de catálogo tiradas separadamente, espere similaridades menores e calibre o limite pela demo web.

## Vídeos: cada produto contado uma vez

![skuscope video num corredor de supermercado visto do carrinho: os produtos da prateleira da esquerda marcados e coloridos por produto, com a contagem acumulada de produtos vistos e contados](docs/aisle-tracking.gif)
<sub>Saída do `skuscope video` (parte esquerda do quadro) em 4,5 s de um [vídeo de banco de imagens](https://www.pexels.com/video/shopping-aisle-perspective-on-supermarket-essentials-29376327/) (Pexels). Catálogo de 12 produtos, 51 fotos tiradas de outros quadros do mesmo vídeo.</sub>

Numa foto, contar é somar os rótulos. Num vídeo, o mesmo produto aparece em dezenas de quadros, então não dá para simplesmente somar os quadros. O `skuscope video` acompanha cada produto de um quadro para o outro e conta cada um uma vez só.

O rastreamento não usa nenhum modelo: é visão computacional clássica (OpenCV) e algumas regras.

1. **Movimento da câmera.** Entre dois quadros, pontos marcantes (ORB) são casados e uma homografia (RANSAC) descreve como a imagem se moveu.
2. **Casamento.** Cada produto já rastreado é deslocado por essa homografia e depois casado com as caixas novas pela sobreposição (IoU). Sem esse passo, uma câmera que anda 40 px por quadro deixa quase nenhuma sobreposição entre quadros, e os produtos ganham identidades novas.
3. **Falhas curtas.** Um produto perdido por causa do borrão por até 30 quadros (1 s) é reencontrado onde o movimento da câmera o levou, com a mesma identidade.
4. **Nome estável.** Cada produto guarda uma média dos seus scores por SKU ao longo dos quadros, então um quadro borrado não troca o nome dele.
5. **Contagem.** Um produto é contado depois de receber nome em 3 quadros, uma vez só, com o nome que tinha naquele momento.

```bash
skuscope video data/shelves/corredor.mp4 --threshold 0.65 --margin 0.05 --motion-region 0 0.5
```

```
135 frames, 1099 different products seen, 415 counted → outputs/corredor_skuscope.mp4
  cup-noodles                      88
  case-beige                       87
  cup-noodles-family               62
  top-ramen-pack                   48
  ...
```

Este vídeo tem muito borrão de movimento, por isso usa um limite menor e uma margem maior que os padrões. `cup-noodles-family` é uma linha de produto (veja [linhas de produto](#5-opcional-linhas-de-produto)).

Uma homografia descreve um plano. Andando por um corredor, a prateleira da esquerda, a da direita e o chão se movem de jeitos diferentes, então `--motion-region 0 0.5` estima o movimento só pela metade esquerda do quadro. A tabela conta os produtos da prateleira da esquerda nos mesmos 4,5 s, rodando o rastreamento sobre as detecções salvas pelos scripts do vídeo de demonstração (recortes do original em 4K, por isso mais produtos recebem nome do que na execução em 1920 px acima):

| Rastreamento | Produtos contados | Duplicatas suspeitas |
|---|---|---|
| Só IoU, sem o movimento da câmera | 607 | 15% |
| Movimento da câmera pelo quadro inteiro | 525 | 12% |
| **Movimento da câmera pela metade esquerda** (`--motion-region 0 0.5`) | **471** | **5%** |

Uma duplicata suspeita é um produto contado que aparece onde o movimento da câmera prevê um produto de mesmo nome, contado antes e depois perdido. É uma estimativa: ninguém contou à mão os produtos únicos do vídeo.

### Acurácia em vídeo, contra quadros conferidos à mão

| Cena | Produtos com o nome certo | Nome errado |
|---|---|---|
| Parede de laticínios, 35 produtos, 5 ou 6 fotos de cada | 174 de 196 (89%) | 10 |
| Corredor de macarrão (o vídeo acima), 12 produtos | 129 de 167 (77%), 45 deles pela linha de produto | 7 |

Os erros que sobram são entre produtos parecidos: cream cheese e cottage cheese da mesma marca, ricota e mascarpone. Esses números são otimistas, porque as fotos do catálogo saíram de outros quadros das mesmas filmagens, com a mesma luz e o mesmo ângulo. Foram medidos com os scripts do vídeo de demonstração, que usam este mesmo método.

## Como usar

Testado num MacBook com Apple M2 Pro (16 GB). O DINO roda na GPU do Mac pelo MPS. Linux e Windows também devem funcionar, usando CPU ou CUDA.

```bash
git clone https://github.com/escudero/skuscope.git
cd skuscope
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[app]"
```

Os pesos dos modelos são baixados no primeiro uso, cerca de 200 MB no total (YOLO11s e DINOv2-small).

Para reproduzir antes os resultados acima, os catálogos e o passo a passo estão em [`examples/`](examples/README.pt-BR.md).

### 1. Fotografe seu catálogo

Escolha de 5 a 10 produtos e tire **de 3 a 5 fotos de cada**, com um produto por foto e alguns ângulos diferentes. Salve uma pasta por SKU:

```
data/catalog/
├── coca-cola-350ml/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.heic        ← fotos de iPhone funcionam direto
├── guarana-antarctica-350ml/
│   └── ...
└── nescau-400g/
    └── ...
```

### 2. Monte o índice do catálogo

```bash
skuscope build-catalog data/catalog
```

O YOLO recorta cada foto no maior produto encontrado, o DINO transforma o recorte em vetor, e o índice é salvo em `index/`.

### 3. Reconheça uma gôndola

```bash
skuscope recognize data/shelves/minha-gondola.jpg --json outputs/minha-gondola.json
```

```
42 products detected → outputs/minha-gondola_skuscope.jpg
  coca-cola-350ml                  12
  guarana-antarctica-350ml          9
  unknown                          21
```

### 4. Reconheça um vídeo

```bash
skuscope video data/shelves/meu-corredor.mp4
```

Ele grava o vídeo anotado em `outputs/meu-corredor_skuscope.mp4` e mostra quantos produtos diferentes viu e contou. Não é em tempo real: num M2 Pro ele processa cerca de 0,4 quadro por segundo em 1920 px.

### 5. Opcional: linhas de produto

Variantes de uma mesma linha (sabores, tamanhos, versões light) podem ser quase idênticas na prateleira. Liste-as em `data/catalog/families.json`:

```json
{
  "cup-noodles": ["cup-noodles-galinha", "cup-noodles-carne", "cup-noodles-camarao"]
}
```

Quando a dúvida é só entre esses irmãos, o produto recebe o nome da linha (`cup-noodles`) em vez de um sabor chutado, ou de "unknown". Rode o `build-catalog` de novo depois de mudar o arquivo.

### 6. Demo web

```bash
skuscope app
```

Abre uma página Gradio onde você envia uma foto e ajusta ao vivo o limite de similaridade, a margem e a confiança da detecção.

## Ajustes

| Opção | Padrão | O que faz |
|---|---|---|
| `--threshold` | `0.8` | Similaridade mínima para aceitar um SKU. Aumente para errar menos o rótulo; diminua para ter menos "unknown". Com o DINOv2, até produtos diferentes ficam entre 0,6 e 0,8. |
| `--margin` | `0.03` | Quanto o SKU mais parecido precisa superar o segundo colocado. Aumente para errar menos o nome de produtos parecidos; `0` desliga. |
| `--conf` | `0.25` | Confiança mínima da detecção do YOLO. |
| `--detector` | `s` | `n` é mais rápido; `s` é mais preciso. |
| `--embedder` | `dinov2-small` | `dinov2-base` é mais preciso e mais lento. `dinov3-small` ainda não foi testado e tem acesso controlado no Hugging Face: aceite a licença e rode `hf auth login` antes. |
| `--no-crop` | desligado | Usa a foto inteira do catálogo em vez do maior produto detectado. |
| `--motion-region` | `0 1` | Só no `video`: estima o movimento da câmera por esta faixa da largura, em frações. Use o lado da prateleira que você quer contar. |

## Limitações

- **Variantes quase idênticas** ("Original" e "Zero", 350 ml e 600 ml) podem mudar só no texto impresso. Só com visão é difícil diferenciar; o próximo passo natural é adicionar OCR.
- **Gôndolas densas em alta resolução:** o detector trabalha em 640 px, então produtos muito pequenos em fotos muito largas podem passar despercebidos. Recorte a foto ou fotografe mais de perto.
- **Produtos tapados ou virados de costas** são difíceis para qualquer abordagem.
- **Produtos que não estão no catálogo** podem ter similaridade alta e receber um rótulo errado (veja os resultados acima).
- **Fotos de catálogo em close:** o detector foi treinado em gôndolas densas. Numa foto de um único produto, ele muitas vezes não encontra nada, ou encontra só um pedaço do rótulo. Quando a maior caixa cobre menos de 20% da foto, o SkuScope usa a foto inteira.
- O limite de similaridade depende dos seus produtos e fotos. Calibre pela demo web.
- **Vídeo é lento:** cada quadro passa pelo YOLO e pelo DINO, cerca de 0,4 quadro por segundo num M2 Pro. Movimentos rápidos, com muito borrão, perdem produtos por mais de 1 s e os contam de novo.

## Próximos passos

- [ ] Comparar DINOv2, DINOv3 e CLIP no dataset [RP2K](https://www.pinlandata.com/rp2k_dataset/)
- [ ] OCR para diferenciar variantes com a mesma embalagem
- [ ] Verificação de planograma: comparar a sequência reconhecida com o layout esperado
- [ ] Dividir fotos muito largas em blocos (*tiling*)
- [x] Margem entre os dois SKUs mais parecidos, e linhas de produto para variantes parecidas
- [x] Vídeo: seguir o movimento da câmera e contar cada produto uma vez
- [ ] Detectar melhor o "unknown" com fotos "distratoras" de produtos que você não vende
- [ ] Vídeo mais rápido: detectar a cada poucos quadros e deixar o rastreamento preencher os intervalos
- [ ] Busca aproximada (FAISS) para catálogos com milhões de fotos. No macOS, o FAISS do pip e o PyTorch trazem cada um o seu `libomp`, e isso derruba o processo.

## Modelos e licenças

| Componente | Origem | Licença |
|---|---|---|
| Código do SkuScope | este repositório | [AGPL-3.0](LICENSE). O Ultralytics YOLO é AGPL-3.0. |
| Detector | [chistopat/sku110k-yolo11-object-detector](https://huggingface.co/chistopat/sku110k-yolo11-object-detector) (ONNX, revisão fixada, SHA-256 verificado) | Segue os [termos do dataset SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) (uso em pesquisa) |
| Embeddings | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) | Apache-2.0 |
| Foto de exemplo | [Soft drink shelf](https://commons.wikimedia.org/wiki/File:Soft_drink_shelf.JPG), Wikimedia Commons | Domínio público |
| Vídeo de exemplo | [Shopping aisle perspective](https://www.pexels.com/video/shopping-aisle-perspective-on-supermarket-essentials-29376327/), Pexels | [Licença Pexels](https://www.pexels.com/license/) |

O detector é carregado pela versão ONNX, e não pelo checkpoint `.pt`. Carregar um `.pt` de terceiros pode executar código arbitrário (pickle); carregar ONNX não.

## Créditos e referências

- Goldman et al., *Precise Detection in Densely Packed Scenes* (CVPR 2019). É o dataset SKU-110K.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (2023)
- *Real-time retail planogram compliance application using computer vision and virtual shelves*, Scientific Reports. Reconhecimento de SKU com poucas fotos nas lojas 7-Eleven de Taiwan. [link](https://www.nature.com/articles/s41598-025-27773-5)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## Desenvolvimento

```bash
pip install -e ".[dev]"
pytest
```
