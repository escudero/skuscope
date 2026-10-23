# Examples

These are the exact catalogs I used for the results in the [main README](../README.md). Running one of them first is the quickest way to see SkuScope working, and to have something to compare against, before you photograph your own products.

🇧🇷 [Leia em português](README.pt-BR.md)

There are three, from simplest to hardest:

| Example | What you'll see | Catalog |
|---|---|---|
| [Soft drinks](#1-soft-drinks-start-here) | One shelf photo, every bottle labelled | 7 products, 1 photo each |
| [Noodle aisle](#2-noodle-aisle-counting-in-a-video) | A cart rolling down an aisle, each product counted once | 12 products, 51 photos |
| [Dairy wall](#3-dairy-wall-a-crowded-shelf) | Almost 300 products in a single frame | 35 products, 201 photos |

A heads-up before you start: none of these products are Brazilian. The soft drinks are Australian and the dairy wall is from New Zealand. That's fine, because what you're testing is the method, not the brands. When you want to try your own products, take 3 to 5 photos of each one, put them in one folder per product, and run `build-catalog`. No training.

And to be upfront about the numbers: they are optimistic. I cropped every catalog photo from the same footage it is tested on, so the light and the angle are the same. With photos taken separately, expect lower scores.

## Before you start

Install SkuScope as described in the [Quick start](../README.md#quick-start). Examples 2 and 3 download a video, so they also need `curl` and `ffmpeg` (on macOS, `brew install ffmpeg`). Run every command from the root of the repository.

If you get `command not found: skuscope`, the virtual environment isn't active in this terminal. Run `source .venv/bin/activate` and try again.

## 1. Soft drinks: start here

This one needs nothing beyond the repository. The shelf photo is already in the folder.

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

Open `outputs/shelf_skuscope.jpg`. It should match the [example image](../docs/example-result.jpg) exactly: 30 known bottles with the right name, and 1 wrong label.

That wrong label is the interesting part. I left two brands out of the catalog on purpose, Traditional's Lemonade and Cream Soda, to see what the model does with products it has never seen. Most of them end up as `unknown` (the grey boxes), which is what you want. One Lemonade still gets called `mountain-dew`.

Now play with the two knobs and watch the counts change:

- `--threshold 0.85`: stricter. No more wrong labels, but fewer known bottles get a name.
- `--margin 0`: turns off the check that the best match must clearly beat the second best. Wrong labels come back.

## 2. Noodle aisle: counting in a video

In a photo, counting is just adding up labels. In a video, the same product shows up in dozens of frames, so SkuScope follows each one from frame to frame and counts it once.

First, download the [video](https://www.pexels.com/video/shopping-aisle-perspective-on-supermarket-essentials-29376327/) from Pexels (65 MB, 4K). Then cut the same 4.5 seconds I used, at 1920 px:

```bash
mkdir -p data/shelves
curl -L -A "Mozilla/5.0" -o data/shelves/aisle-4k.mp4 https://www.pexels.com/download/video/29376327/
ffmpeg -ss 1.6 -i data/shelves/aisle-4k.mp4 -t 4.5 -vf scale=1920:-2 -an data/shelves/corredor.mp4
```

The catalog photos are already tight crops of each product, so build the catalog with `--no-crop`. Otherwise the detector tries to find a product inside the crop and often misses it.

```bash
skuscope build-catalog examples/noodle-aisle/catalog --index examples/noodle-aisle/index --no-crop
skuscope video data/shelves/corredor.mp4 --index examples/noodle-aisle/index \
  --threshold 0.65 --margin 0.05 --motion-region 0 0.5
```

Go grab a coffee. It takes a few minutes, because every frame goes through YOLO and DINO (about 0.4 frames per second on my M2 Pro).

```
135 frames, 1075 different products seen, 410 counted → outputs/corredor_skuscope.mp4
  case-beige                       88
  cup-noodles                      86
  top-ramen-pack                   66
  cup-noodles-family               50
  ...
```

The main README says 1099 seen and 415 counted. Your numbers will be close but may not match exactly. Cutting and re-encoding the clip shifts a few detections, and that's enough to move the counts a little.

Why these options:

- **`--threshold 0.65 --margin 0.05`:** the cart moves and the video is blurry, so the similarities are lower than on a still photo. A lower threshold lets products get a name, and a bigger margin keeps lookalikes from getting the wrong one.
- **`--motion-region 0 0.5`:** when you walk down an aisle, the left shelf, the right shelf and the floor all move differently. This tells SkuScope to estimate the camera motion from the left half only, the shelf we're counting.
- **`catalog/families.json`:** some products are nearly identical on the shelf, like the three kinds of Cup Noodles. When SkuScope can't tell them apart, it names the product line (`cup-noodles-family`) instead of guessing a flavour.

## 3. Dairy wall: a crowded shelf

This is the hardest one: a whole wall of yoghurt, sour cream and cheese, with many products that look alike.

Download the [video](https://www.pexels.com/video/dairy-aisle-shopping-in-supermarket-39221961/) from Pexels (6 MB) and grab the frame at 5 seconds:

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

207 of the 296 products get a name. If you open the image, you'll see most of the `unknown` boxes are on the right side of the wall. Those are cheeses I didn't put in the catalog, so `unknown` is the right answer for them.

None of the catalog photos come from this frame. I took them from other moments of the same video (between 1 and 10 seconds), so the model is never matching a product against its own picture.

About the 89% accuracy in the main README: I got it by labelling this frame by hand and comparing, using the scripts from the demo video. The command above uses the same method, but the hand labels aren't included here, so it can't tell you how many it got right. You'll have to look.

Want to see the whole wall go by? Run `skuscope video data/shelves/dairy.mp4` with the same index and options.

## Where the images come from

| What | Source | License |
|---|---|---|
| `soft-drinks/shelf.jpg` and its catalog crops | [Soft drink shelf](https://commons.wikimedia.org/wiki/File:Soft_drink_shelf.JPG), Wikimedia Commons, scaled to 1600 px | Public domain |
| `noodle-aisle/catalog` | Crops from [Shopping aisle perspective](https://www.pexels.com/video/shopping-aisle-perspective-on-supermarket-essentials-29376327/), Pexels | [Pexels license](https://www.pexels.com/license/) |
| `dairy-wall/catalog` | Crops from [Dairy aisle shopping](https://www.pexels.com/video/dairy-aisle-shopping-in-supermarket-39221961/), Pexels | [Pexels license](https://www.pexels.com/license/) |

The videos aren't in the repository. You download them straight from Pexels.

If you try SkuScope with your own products, I'd love to hear how it went. Open an issue, or find me on LinkedIn.
