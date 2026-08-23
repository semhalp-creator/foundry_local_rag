# Proje Günlüğü — Foundry Local RAG Assistant

Bu dosya, `Summer School Foundry Local Plan.docx` planını takip ederek bu projede hafta hafta ne yaptığımızın ayrıntılı bir kaydı. `README.md` "projeyi nasıl çalıştırırım" sorusuna cevap veriyor; bu dosya ise "buraya nasıl geldik" sorusuna cevap veriyor.

---

## Başlangıç noktası: main.py'deki bug

Elimizde zaten Hafta 1-2 seviyesinde yazılmış bir `main.py` vardı — bellek-içi bir döküman listesi, cosine similarity ile retrieval, ve streaming bir chat cevabı. Bu dosya çalışıyordu ama bir hatası vardı:

```python
for chunk in chat_client.complete_streaming_chat(messages):
    content = chunk.choices[0].delta.content   # önce indeksleniyor
    if not chunk.choices:                       # boş kontrolü sonra yapılıyor
        continue
```

`chunk.choices` bazı stream chunk'larında boş geliyordu (örn. stream'in son chunk'ı), ve indeksleme boş kontrolünden **önce** yapıldığı için `IndexError: list index out of range` hatası veriyordu. Sıra değiştirilerek düzeltildi:

```python
for chunk in chat_client.complete_streaming_chat(messages):
    if not chunk.choices:
        continue
    content = chunk.choices[0].delta.content
```

Bu düzeltmeden sonra `main.py`'nin tamamını satır satır, fonksiyon fonksiyon inceledik: `cosine_similarity` (dot product / norm formülü), `find_relevant` (`enumerate`, `sort(key=lambda ...)`, slicing), SDK başlatma, embedding/chat model yükleme, ve `role: system` vs `role: user` ayrımıyla temel prompt engineering kavramları.

## Planı bulma

`~/Downloads/Summer School Foundry Local Plan.docx` dosyasını bulduk — 6 haftalık, 3 fazlı bir müfredat:

- **Faz 1 (Hafta 1-2):** Temel öğrenme — RAG kavramı, Foundry Local kurulumu, embedding'ler, SQLite
- **Faz 2 (Hafta 3-4):** Proje implementasyonu — ingestion pipeline, retrieval, LLM entegrasyonu
- **Faz 3 (Hafta 5-6):** Test, değerlendirme, dokümantasyon, final sunum

Mevcut `main.py`'nin Hafta 1-2'nin "embedding demo" egzersiziyle birebir örtüştüğünü tespit ettik — yani zaten oradaydık, sıradaki adım Hafta 2'nin SQLite kısmıydı.

---

## Hafta 2 — SQLite Mekaniği

**Dosya:** [`sqlite_practice.py`](sqlite_practice.py)

Planın "SQL sandbox" egzersizi: `documents(id, content, embedding)` şemalı basit bir SQLite tablosu oluşturup birkaç örnek satır ekleyip (`INSERT`), id'ye ve anahtar kelimeye göre sorgulayıp (`SELECT ... WHERE`) mekaniği öğrenme amaçlı, RAG pipeline'ından bağımsız bir pratik. Test edildi, `practice.db` oluştu, doğru çalıştığı doğrulandı.

---

## Hafta 3 — Ingestion & Retrieval

**Dosyalar:** [`ingest.py`](ingest.py), [`retrieval.py`](retrieval.py)

İlk versiyon sadece `main.py`'deki 8 cümleyi embed edip SQLite'a yazıyordu, ama plan **chunking** (dökümanı parçalara ayırma) ve **doğrulama testi** de istiyordu — ikisi de eksikti, sonradan eklendi:

- `chunk_text()` fonksiyonu — bir dökümanı paragraf sınırlarına göre parçalara ayırıyor. Bunu göstermek için 3 paragraflık gerçekçi bir örnek döküman (`FOUNDRY_LOCAL_OVERVIEW`) eklendi.
- Ingestion sonrası `SELECT COUNT(*)` ile satır sayısını bekleneni ile karşılaştıran bir doğrulama eklendi (`"Ingested and verified 11 chunks in rag.db."`)
- `retrieval.py`'de `get_top_chunks(query, ...)` fonksiyonu — SQLite'tan tüm embedding'leri çekip `main.py`'deki `cosine_similarity`'yi **import ederek** (kod tekrarı yapmadan) benzerlik hesaplıyor.

Test: birkaç örnek soruyla `get_top_chunks` çalıştırıldı, en alakalı dökümanların doğru şekilde en yüksek skoru aldığı doğrulandı.

---

## Hafta 4 — LLM Entegrasyonu

**Dosya:** [`app.py`](app.py)

- **Model yükseltmesi:** Foundry Local kataloğu incelendi, `qwen2.5-0.5b` (0.5B, çok küçük) yerine planın önerdiği **`phi-3.5-mini`** (~3.8B) kullanılmaya başlandı.
- **`answer_query()`** fonksiyonu yazıldı: `get_top_chunks()` ile bağlam çek → grounded system prompt kur → streaming cevap üret.
- **CLI arayüzü (Option A)** — planın "zaman kısıtı içinde tamamlanmayı garantileyen" önerdiği en basit seçenek.
- **Responsible outputs düzeltmesi:** İlk testte, kapsam dışı bir soruda ("Fransa'nın başkenti nedir?") model "bilmiyorum" dedikten sonra kendi ezberinden ekstra bilgi ekliyordu (*"Paris'tir, General Knowledge"*). System prompt'a *"context dışı bilgi ekleme, öneri olarak bile"* talimatı eklenerek kapatıldı.

---

## Hafta 5 — Test, Performans, Değerlendirme

**Dosyalar:** [`test_suite.py`](test_suite.py), [`test_results.md`](test_results.md)

Planın 3 alt başlığının hepsi ayrı ayrı karşılandı:

1. **Functional Testing** — cevaplanabilir + cevaplanamaz + genel soru karışımından oluşan otomatik test seti, her sorgu için pass/fail, süre, chunk sayısı ölçülüyor.
2. **Performance & Debugging** — yanıt süreleri ölçüldü (~1-3s hedefinin içinde), embedding'lerin tekrar hesaplanmadığı (SQLite'ta cache'li) doğrulandı, formatting/retrieval sağlık kontrolleri eklendi.
3. **Evaluation & Improvement (self-critique)** — planın sorduğu üç soruya (doğru mu? öz mü? kaynak gösteriyor mu?) otomatik, verilere dayalı cevaplar üretildi.

**Edge case düzeltmesi:** Plan "empty query input" test etmeyi istiyordu. Test edilince görüldü ki boş Enter, `"quit"` ile aynı davranıyordu (programdan çıkıyordu). Bu, kullanıcı dostu olmadığı için düzeltildi — artık boş Enter tekrar soru soruyor, sadece `"quit"` çıkış yapıyor.

### Hafta 4/5 stretch goal — Option C: Web arayüzü

**Dosyalar:** [`web_app.py`](web_app.py), [`templates/index.html`](templates/index.html)

Planın "Options B/C can be offered ... as stretch goals for Week 5 if time remains" notuna dayanarak, Flask tabanlı basit bir web arayüzü eklendi (`/ask` JSON endpoint'i, `answer_query()`'i CLI ile birebir aynı şekilde kullanıyor). İlk versiyon oldukça "gösterişli" (sohbet balonları, avatarlar, animasyonlar) yapıldı — sonra fark edildi ki plan açıkça **"Basic HTML+JS UI"** ve **"minimal UI elements"** diyordu. Kullanıcının "biz basic işinde patladık" tespiti üzerine, sade bir versiyona (tek text box + buton + cevap alanı) geri döndürüldü — planın gerçek beklentisine uygun hale getirildi.

---

## Hafta 6 — Dokümantasyon & Sunum Hazırlığı

**Dosyalar:** [`README.md`](README.md), [`PRESENTATION.md`](PRESENTATION.md)

- **README.md** — projenin amacı, RAG pipeline'ının nasıl çalıştığı, dosya yapısı, kurulum/kullanım talimatları, tasarım kararları & bilinen sınırlamalar, referanslar.
- **Kod temizliği** — tüm `.py` dosyaları debug print / TODO / commented-out kod için tarandı (temiz çıktı), küçük bir stil düzeltmesi yapıldı (gereksiz f-string).
- **PRESENTATION.md** — planın istediği 4 başlık: Problem Statement, Key Features, Live Demo script, Lessons Learned.

**Kritik an — prova gerçek bir bug'ı ortaya çıkardı:** Plan sadece "sunum notu yaz" değil, **"rehearsed"** (prova edilmiş) diyordu. `PRESENTATION.md`'deki demo script'i gerçekten baştan sona çalıştırınca, daha önce "çözüldü" sanılan bir hata **CLI'nin canlı ekran çıktısında hâlâ görünüyordu**: kapsam dışı bir soruda model, talimata rağmen uydurma bir kaynak satırı ekliyordu, ve önceki düzeltmemiz sadece fonksiyonun *döndürdüğü* değeri temizliyordu — ekrana zaten basılmış olan streaming metni değil. Çözüm: CLI artık cevabı **önce tamamen toplayıp temizliyor, sonra ekrana basıyor** (streaming yerine buffered) — böylece ekran çıktısı ile döndürülen değer her zaman birebir aynı.

---

## Hafta 6 sonrası — Gerçek Kullanıcı Testi (en değerli kısım)

Kullanıcı projeyi bizzat kullanarak, planın öngörmediği birkaç gerçek hata daha buldu. Her biri klasik bir döngüyle kapatıldı: **keşif → kök neden analizi → düzeltme → doğrulama → kalıcı regresyon testi.**

### 1. Model seçimi doğru mu?

Kontrol edildi: `qwen3-embedding-0.6b` ve `phi-3.5-mini`, planın verdiği örneklerle birebir eşleşiyor. Planın "hız önceliği" notu ile "3-5B model" önerisi arasındaki görünür çelişki, ölçtüğümüz gerçek performans verisiyle (~2s/soru) çözülmüş oldu.

### 2. "Hız senin dediğinden yavaş" — iki gizli sebep bulundu

- **Model yükleme süresi sayılmamıştı:** `app.py`/`web_app.py` her başlatıldığında modellerin belleğe yüklenmesi **~14.3 saniye** sürüyor — bu, "soru başına 2 saniye" rakamına dahil değildi.
- **Buffered streaming, algılanan hızı düşürdü:** Hafta 6'daki düzeltme (canlı akış yerine tam cevabı bekleyip basma) toplam süreyi değiştirmedi ama kullanıcının "bir şeyler oluyor" hissini kaldırdı.

### 3. Türkçe, kapsam dışı soru → uydurma cevap

Kullanıcı web arayüzünden `"bu gece hava kaç derece"` diye sordu. Model, zayıf/alakasız retrieval sonuçlarıyla (skor ~0.45-0.47) karşılaşınca, İngilizce yazılmış "bilmiyorsan söyle" talimatını Türkçe cevap üretirken düzgün uygulayamadı — anlamsız, kısmen uydurma bir metin üretti.

**Kök neden:** Asıl sorun dil değil, **düşük alaka skorlu chunk'ların hiç modele gönderilmemesi gerektiğiydi.**

**Düzeltme:** `app.py`'ye `MIN_RELEVANT_SCORE = 0.55` eşiği eklendi — en iyi chunk bile bu eşiğin altındaysa, model **hiç çağrılmadan** sabit bir `NO_MATCH_ANSWER` dönülüyor. Dilden bağımsız, daha hızlı (0.03s), hallüsinasyon riski sıfır. Bu senaryo `test_suite.py`'ye **kalıcı bir regresyon testi** olarak eklendi (8. test case).

### 4. "Bilmiyorum" cevabında da kaynak gösterme mantıksızlığı

Kullanıcı sordu: *"bilmiyorum dediğinde de kaynak vermesi mantıklı mı?"* — haklı çıktı. `answer_query()`, eşik-altı durumda reddedilen chunk'ları hâlâ döndürüyordu, bu da web arayüzünde "I don't have information..." cevabının altında yanlışlıkla "Sources: Foundry Local FAQ" gösterilmesine yol açıyordu. Düzeltme: eşik-altı durumda `results` yerine boş liste (`[]`) döndürülüyor.

### 5. Boş "Sources:" etiketi

Chunks boş olduğunda bile `#sources` div'i `"Sources: "` (hiçbir şey olmadan) yazıyordu. JS'te `sources.length` kontrolü eklenerek boşsa hiç yazı basılmaması sağlandı.

### 6. Çift kaynak gösterimi + "hi" testi

Kullanıcı `"hi"` yazınca hem modelin kendi ürettiği `"Source: Foundry Local FAQ"` (cevap metninin içinde) hem bizim kod tarafında ürettiğimiz `"Sources: Foundry Local FAQ"` (ayrı div) **aynı anda** görünüyordu — çift, tutarsız bir gösterim. Ayrıca `"hi"` gibi bir selamlama bile eşiği (0.55) tesadüfen geçebiliyordu.

**Kök çözüm:** İki bağımsız kaynak-gösterme mekanizması **tek bir deterministik mekanizmaya** indirildi:
- System prompt'tan modele "kaynağını yaz" talimatı tamamen kaldırıldı
- Yeni `format_source_line(results)` fonksiyonu — CLI ve web'in **ikisi de aynı veriden** (gerçekten kullanılan `results`) besleniyor, modelin kendi beyanına hiç güvenilmiyor

**Sonuç:** 8/8 test geçiyor, CLI ve web'de artık tek, tutarlı, her zaman doğru bir kaynak satırı var.

---

## Genel özet

| Hafta | Ana çıktı | Dosyalar |
|---|---|---|
| 1-2 | Bug fix + kod açıklaması, plan keşfi | `main.py` |
| 2 | SQLite mekaniği pratiği | `sqlite_practice.py` |
| 3 | Chunking + ingestion + SQLite retrieval | `ingest.py`, `retrieval.py` |
| 4 | LLM entegrasyonu, model yükseltme, CLI | `app.py` |
| 5 | Otomatik test, performans, edge case, web UI (stretch) | `test_suite.py`, `web_app.py`, `templates/index.html` |
| 6 | README, kod temizliği, sunum notları, canlı prova bug'ı | `README.md`, `PRESENTATION.md` |
| 6+ | Gerçek kullanıcı testiyle bulunan 6 gerçek hata, hepsi düzeltildi ve kalıcı test edildi | `app.py`, `test_suite.py` |

Planın 6 haftalık müfredatının tamamı (Foundational Learning → Project Implementation → Testing, Evaluation & Documentation) uygulandı, ve buna ek olarak Hafta 6 sonrasında yapılan gerçek kullanıcı testi, planın öngörmediği ama gerçek dünyada karşılaşılacak türden **6 ayrı, somut hatayı** ortaya çıkarıp düzeltti.
