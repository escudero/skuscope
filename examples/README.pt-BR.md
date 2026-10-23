# Exemplos

Estes são os mesmos catálogos que usei nos resultados do [README principal](../README.pt-BR.md). Rodar um deles primeiro é o jeito mais rápido de ver o SkuScope funcionando, e de ter com o que comparar, antes de fotografar os seus produtos.

🇺🇸 [Read in English](README.md)

São três, do mais simples ao mais difícil:

| Exemplo | O que você vai ver | Catálogo |
|---|---|---|
| [Refrigerantes](#1-refrigerantes-comece-por-aqui) | Uma foto de gôndola, com cada garrafa identificada | 7 produtos, 1 foto de cada |
| [Corredor de macarrão](#2-corredor-de-macarrão-contando-num-vídeo) | Um carrinho passando pelo corredor, com cada produto contado uma vez | 12 produtos, 51 fotos |
| [Parede de laticínios](#3-parede-de-laticínios-uma-gôndola-lotada) | Quase 300 produtos num quadro só | 35 produtos, 201 fotos |

Um aviso antes de começar: nenhum desses produtos é brasileiro. Os refrigerantes são australianos, e a parede de laticínios é da Nova Zelândia. Isso não atrapalha, porque o que você está testando é o método, não as marcas. Quando quiser testar com os seus produtos, tire de 3 a 5 fotos de cada um, coloque numa pasta por produto e rode o `build-catalog`. Sem treino.

E, para ser transparente sobre os números: eles são otimistas. Recortei cada foto do catálogo da mesma filmagem em que ela é testada, então a luz e o ângulo são os mesmos. Com fotos tiradas à parte, espere similaridades menores.

## Antes de começar

Instale o SkuScope como em [Como usar](../README.pt-BR.md#como-usar). Os exemplos 2 e 3 baixam um vídeo, então também precisam de `curl` e `ffmpeg` (no macOS, `brew install ffmpeg`). Rode todos os comandos a partir da raiz do repositório.

Se aparecer `command not found: skuscope`, o ambiente virtual não está ativo neste terminal. Rode `source .venv/bin/activate` e tente de novo.

## 1. Refrigerantes: comece por aqui

Este não precisa de nada além do repositório. A foto da gôndola já está na pasta.

```bash
skuscope build-catalog examples/soft-drinks/catalog --index examples/soft-drinks/index
skuscope recognize examples/soft-drinks/shelf.jpg --index examples/soft-drinks/index
```

```
48 products detected → outputs/shelf_skuscope.jpg
  unknown                          17
  schweppes-lemonade                6
  pasito                            5
  solo                              5
  sunkist                           5
  passiona                          4
  mountain-dew                      3
  7up                               3
```

Abra o `outputs/shelf_skuscope.jpg`. Ele deve ficar igual à [imagem de exemplo](../docs/example-result.jpg): 30 garrafas conhecidas com o nome certo e 1 com o nome errado.

Esse nome errado é a parte interessante. Deixei duas marcas fora do catálogo de propósito, a Traditional's Lemonade e a Cream Soda, para ver o que o modelo faz com produtos que nunca viu. A maioria vira `unknown` (as caixas cinza), que é o que a gente quer. Mas uma Lemonade ainda é chamada de `mountain-dew`.

Agora mexa nos dois ajustes e veja a contagem mudar:

- `--threshold 0.85`: mais exigente. Nenhum nome errado, mas menos garrafas conhecidas recebem nome.
- `--margin 0`: desliga a regra de que o produto mais parecido precisa ganhar do segundo com folga. Os nomes errados voltam.

## 2. Corredor de macarrão: contando num vídeo

Numa foto, contar é só somar os nomes. Num vídeo, o mesmo produto aparece em dezenas de quadros, então o SkuScope acompanha cada um de quadro em quadro e conta uma vez só.

Primeiro, baixe o [vídeo](https://www.pexels.com/video/shopping-aisle-perspective-on-supermarket-essentials-29376327/) do Pexels (65 MB, 4K). Depois, corte os mesmos 4,5 segundos que eu usei, em 1920 px:

```bash
mkdir -p data/shelves
curl -L -A "Mozilla/5.0" -o data/shelves/aisle-4k.mp4 https://www.pexels.com/download/video/29376327/
ffmpeg -ss 1.6 -i data/shelves/aisle-4k.mp4 -t 4.5 -vf scale=1920:-2 -an data/shelves/corredor.mp4
```

As fotos do catálogo já são recortes justos de cada produto, então monte o catálogo com `--no-crop`. Sem isso, o detector tenta achar um produto dentro do recorte e muitas vezes não acha.

```bash
skuscope build-catalog examples/noodle-aisle/catalog --index examples/noodle-aisle/index --no-crop
skuscope video data/shelves/corredor.mp4 --index examples/noodle-aisle/index \
  --threshold 0.65 --margin 0.05 --motion-region 0 0.5
```

Vá pegar um café. Leva alguns minutos, porque cada quadro passa pelo YOLO e pelo DINO (cerca de 0,4 quadro por segundo no meu M2 Pro).

```
135 frames, 1075 different products seen, 410 counted → outputs/corredor_skuscope.mp4
  case-beige                       88
  cup-noodles                      86
  top-ramen-pack                   66
  cup-noodles-family               50
  ...
```

O README principal fala em 1099 vistos e 415 contados. Os seus números vão ficar perto, mas podem não bater exatamente. Cortar e recodificar o trecho muda algumas detecções, e isso basta para a contagem variar um pouco.

Por que essas opções:

- **`--threshold 0.65 --margin 0.05`:** o carrinho anda e o vídeo fica borrado, então as similaridades ficam menores que numa foto parada. O limite menor deixa os produtos receberem nome, e a margem maior evita que produtos parecidos recebam o nome errado.
- **`--motion-region 0 0.5`:** quando você anda por um corredor, a prateleira da esquerda, a da direita e o chão se movem de jeitos diferentes. Isso diz ao SkuScope para estimar o movimento da câmera só pela metade esquerda, que é a prateleira que estamos contando.
- **`catalog/families.json`:** alguns produtos são quase idênticos na prateleira, como os três tipos de Cup Noodles. Quando o SkuScope não consegue diferenciar, ele dá o nome da linha de produto (`cup-noodles-family`) em vez de chutar um sabor.

## 3. Parede de laticínios: uma gôndola lotada

Este é o mais difícil: uma parede inteira de iogurte, creme azedo e queijo, com muitos produtos parecidos.

Baixe o [vídeo](https://www.pexels.com/video/dairy-aisle-shopping-in-supermarket-39221961/) do Pexels (6 MB) e tire o quadro dos 5 segundos:

```bash
mkdir -p data/shelves
curl -L -A "Mozilla/5.0" -o data/shelves/dairy.mp4 https://www.pexels.com/download/video/39221961/
ffmpeg -ss 5 -i data/shelves/dairy.mp4 -frames:v 1 -q:v 2 data/shelves/dairy.jpg
```

```bash
skuscope build-catalog examples/dairy-wall/catalog --index examples/dairy-wall/index --no-crop
skuscope recognize data/shelves/dairy.jpg --index examples/dairy-wall/index --threshold 0.70 --margin 0.03
```

```
296 products detected → outputs/dairy_skuscope.jpg
  unknown                          89
  mf-cottage-cheese                24
  yoplait-side                     17
  ...
```

207 dos 296 produtos recebem nome. Se abrir a imagem, você vai ver que a maior parte das caixas `unknown` está do lado direito da parede. São queijos que não coloquei no catálogo, então `unknown` é a resposta certa para eles.

Nenhuma foto do catálogo vem deste quadro. Tirei de outros momentos do mesmo vídeo (entre 1 e 10 segundos), então o modelo nunca compara um produto com a própria foto.

Sobre os 89% de acerto do README principal: cheguei nesse número conferindo este quadro à mão e comparando, com os scripts do vídeo de demonstração. O comando acima usa o mesmo método, mas a conferência feita à mão não vem junto, então ele não tem como dizer quantos acertou. Aí é com você: abra a imagem e confira.

Quer ver a parede inteira passando? Rode `skuscope video data/shelves/dairy.mp4` com o mesmo índice e as mesmas opções.

## De onde vêm as imagens

| O quê | Origem | Licença |
|---|---|---|
| `soft-drinks/shelf.jpg` e os recortes do catálogo | [Soft drink shelf](https://commons.wikimedia.org/wiki/File:Soft_drink_shelf.JPG), Wikimedia Commons, reduzida para 1600 px | Domínio público |
| `noodle-aisle/catalog` | Recortes de [Shopping aisle perspective](https://www.pexels.com/video/shopping-aisle-perspective-on-supermarket-essentials-29376327/), Pexels | [Licença Pexels](https://www.pexels.com/license/) |
| `dairy-wall/catalog` | Recortes de [Dairy aisle shopping](https://www.pexels.com/video/dairy-aisle-shopping-in-supermarket-39221961/), Pexels | [Licença Pexels](https://www.pexels.com/license/) |

Os vídeos não estão no repositório. Você baixa direto do Pexels.

Se testar o SkuScope com os seus produtos, vou gostar de saber como foi. Abra uma issue ou me chame no LinkedIn.
