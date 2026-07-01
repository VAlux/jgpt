# Ukrainian training datasets

Char-level datasets for jgpt, in the same format as
`app/src/main/resources/ua-settlement-names.txt`: **UTF-8, one item per line,
uppercase Ukrainian Cyrillic, sorted and deduplicated.**

| File | Entries | Description | Source |
| --- | --- | --- | --- |
| `ua-given-names.txt` | 329 | Ukrainian given names (імена) | en.wiktionary categories |
| `ua-surnames.txt` | 371 | Ukrainian surnames (прізвища) | en.wiktionary category |
| `ua-rivers.txt` | 1028 | Rivers / streams (гідроніми) | GeoNames `UA` (feature STM) |
| `ua-lakes.txt` | 246 | Lakes (озера) | GeoNames `UA` (feature LK*) |
| `ua-mountains.txt` | 250 | Mountains / hills (гори) | GeoNames `UA` (MT/PK/HLL) |
| `ua-railway-stations.txt` | 1471 | Railway stations (станції) | GeoNames `UA` (RSTN/RSTP) |
| `ua-admin-regions.txt` | 1717 | Raions / hromadas (адмінодиниці) | GeoNames `UA` (ADM1-3) |
| `ua-settlements-geonames.txt` | 17252 | Settlements — GeoNames variant | GeoNames `UA` (PPL*) |
| `ua-words.txt` | 333080 | General Ukrainian words | LibreOffice Hunspell `uk_UA.dic` |
| `ua-proverbs.txt` | 963 | Proverbs / sayings (прислів'я, приказки) | uk.wikiquote |

Names/short lists work well at the default `--seq-length 20`. For `ua-words.txt`
(long, and a big vocab) it also works at defaults, but training is slower.

`ua-proverbs.txt` holds full sentences (kept in natural casing, not uppercased),
so raise `--seq-length` to ~80–128 and train longer, e.g.:

```bash
./gradlew run --console=plain -q \
  --args="--data app/data/ua-proverbs.txt --seq-length 100 --iteration-count 3000 --transformer-blocks 2"
```

## Train on one

```bash
./gradlew run --console=plain -q \
  --args="--data app/data/ua-rivers.txt --iteration-count 1000"
```

Or edit `DATA=` in `train.sh` and run `./train.sh`.

## Provenance / licensing

- GeoNames data is CC BY 4.0 (https://www.geonames.org).
- Wiktionary content is CC BY-SA.
- LibreOffice `uk_UA` Hunspell dictionary is GPL/LGPL/MPL.

Ukrainian Cyrillic spellings for GeoNames rows were extracted from the
`alternatenames` blob, preferring tokens with Ukrainian-only letters (і ї є ґ)
and dropping Russian-only ones (ы э ъ ё).
