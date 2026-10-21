# 🔎 SkuScope

**Reconhecimento de SKUs em fotos de gôndola.** O YOLO encontra cada produto na prateleira, e o DINO diz qual SKU cada um é.

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
SKU mais parecido + similaridade (cosseno)
   ↓
similaridade ≥ limite → SKU   |   abaixo → "unknown" (desconhecido)
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

- Com **0,80** (o padrão), 32 das 34 garrafas conhecidas são reconhecidas, mas 5 garrafas desconhecidas recebem um rótulo errado.
- Com **0,85**, nenhuma garrafa desconhecida é rotulada errado, mas 7 garrafas conhecidas viram "unknown".

Esses números são otimistas: os recortes do catálogo vieram da mesma foto, com a mesma luz e o mesmo ângulo. Com fotos de catálogo tiradas separadamente, espere similaridades menores e calibre o limite pela demo web.

## Como usar

Testado num MacBook com Apple M2 Pro (16 GB). O DINO roda na GPU do Mac pelo MPS. Linux e Windows também devem funcionar, usando CPU ou CUDA.

```bash
git clone https://github.com/escudero/skuscope.git
cd skuscope
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[app]"
```

Os pesos dos modelos são baixados no primeiro uso, cerca de 200 MB no total (YOLO11s e DINOv2-small).

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

### 4. Demo web

```bash
skuscope app
```

Abre uma página Gradio onde você envia uma foto e ajusta ao vivo o limite de similaridade e a confiança da detecção.

## Ajustes

| Opção | Padrão | O que faz |
|---|---|---|
| `--threshold` | `0.8` | Similaridade mínima para aceitar um SKU. Aumente para errar menos o rótulo; diminua para ter menos "unknown". Com o DINOv2, até produtos diferentes ficam entre 0,6 e 0,8. |
| `--conf` | `0.25` | Confiança mínima da detecção do YOLO. |
| `--detector` | `s` | `n` é mais rápido; `s` é mais preciso. |
| `--embedder` | `dinov2-small` | `dinov2-base` é mais preciso e mais lento. `dinov3-small` ainda não foi testado e tem acesso controlado no Hugging Face: aceite a licença e rode `hf auth login` antes. |
| `--no-crop` | desligado | Usa a foto inteira do catálogo em vez do maior produto detectado. |

## Limitações

- **Variantes quase idênticas** ("Original" e "Zero", 350 ml e 600 ml) podem mudar só no texto impresso. Só com visão é difícil diferenciar; o próximo passo natural é adicionar OCR.
- **Gôndolas densas em alta resolução:** o detector trabalha em 640 px, então produtos muito pequenos em fotos muito largas podem passar despercebidos. Recorte a foto ou fotografe mais de perto.
- **Produtos tapados ou virados de costas** são difíceis para qualquer abordagem.
- **Produtos que não estão no catálogo** podem ter similaridade alta e receber um rótulo errado (veja os resultados acima).
- **Fotos de catálogo em close:** o detector foi treinado em gôndolas densas. Numa foto de um único produto, ele muitas vezes não encontra nada, ou encontra só um pedaço do rótulo. Quando a maior caixa cobre menos de 20% da foto, o SkuScope usa a foto inteira.
- O limite de similaridade depende dos seus produtos e fotos. Calibre pela demo web.

## Próximos passos

- [ ] Comparar DINOv2, DINOv3 e CLIP no dataset [RP2K](https://www.pinlandata.com/rp2k_dataset/)
- [ ] OCR para diferenciar variantes com a mesma embalagem
- [ ] Verificação de planograma: comparar a sequência reconhecida com o layout esperado
- [ ] Dividir fotos muito largas em blocos (*tiling*)
- [ ] Detectar melhor o "unknown": margem entre os dois SKUs mais parecidos, ou fotos "distratoras" de produtos que você não vende
- [ ] Busca aproximada (FAISS) para catálogos com milhões de fotos. No macOS, o FAISS do pip e o PyTorch trazem cada um o seu `libomp`, e isso derruba o processo.

## Modelos e licenças

| Componente | Origem | Licença |
|---|---|---|
| Código do SkuScope | este repositório | [AGPL-3.0](LICENSE). O Ultralytics YOLO é AGPL-3.0. |
| Detector | [chistopat/sku110k-yolo11-object-detector](https://huggingface.co/chistopat/sku110k-yolo11-object-detector) (ONNX, revisão fixada, SHA-256 verificado) | Segue os [termos do dataset SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) (uso em pesquisa) |
| Embeddings | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) | Apache-2.0 |
| Foto de exemplo | [Soft drink shelf](https://commons.wikimedia.org/wiki/File:Soft_drink_shelf.JPG), Wikimedia Commons | Domínio público |

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
