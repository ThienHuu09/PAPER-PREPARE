# SOICT 2026 Slides: Plain-text Content

Figures are in `images/paper/` (file names given per slide).

---

## Slide 1: Title
**N0N-L1NERS: A Multimodal Video Retrieval System for CPU Execution**

Thien Huu Nguyen (1,4*), Van Vinh Bui (2,4*), Quang Minh Nguyen Tran (1,4*), An Nguyen Hoang (1,4*), Minh Hieu Phan (1,4*), Minh-Huy Tran (3,4†), Nguyet Minh Nguyen Thi (1†), Loi Nguyen Khanh (3,4†)

1. Ngo Quyen High School, Vietnam
2. Tran Dai Nghia High School for The Gifted, Vietnam
3. Ho Chi Minh City University of Technology (HCMUT), VNU-HCM, Vietnam
4. HCMUT EE Machine Learning & IoT Lab, Vietnam

\* Equal contribution; † Corresponding authors

The 15th International Symposium on Information and Communication Technology (SOICT 2026)
December 4, 2026

Footer: N0N-L1NERS Team (HCMUT) | N0N-L1NERS: Video Retrieval on CPU | SOICT 2026

---

## Slide 2: Outline
1. Introduction & Motivation
2. Design & Methodology
3. Results & Case Studies
4. Discussion & Conclusion

Box, "Central research question": How can an interactive system answer ordered multi-event video queries within the response time of a live search, on a CPU only?

Flow: Challenges -> System design -> TRAKE alignment -> Deployment & findings

---

## Slide 3: Challenges in Interactive Video Retrieval
**1. Language**
- Vietnamese queries, English-trained encoders.
- CLIP-style cap: 77 tokens.
- A shared context would use the rest.

**2. Ambiguity**
- Near-duplicate frames confuse similarity.
- Web image search adds delay and cannot confirm a frame is in the collection.

**3. Alignment**
- Naive search over ordered events in one video grows combinatorially.

**4. Latency**
- An external database costs a round trip per query.
- Users issue many queries in succession.

Key message: The challenges are linked: more context costs tokens, richer alignment costs time, and every database access costs latency.

---

## Slide 4: Problem Formulation and Objectives
**Interactive search tasks**
- Target given as text (T-KIS) or image (V-KIS).
- TRAKE: ordered keyframes for up to four events.
- Final round of the HCM City AI Challenge 2026.

**Objective**
- All query types on a CPU only: FAISS for retrieval, Qdrant for keyframe info, no LLM call in the alignment.
- Note: no new alignment algorithm is claimed.

**TRAKE problem**
- Query Q = (y_1, ..., y_L), 1 <= L <= 4, optional context y_0. One keyframe per event, one video nu(.):
  nu(p_1) = ... = nu(p_L), t_{p_1} < ... < t_{p_L}
- Span within T_max = 300 s.

**Contributions**
- System design for order, context and CPU cost.
- TRAKE: DP in seconds + context blending.
- In-memory FAISS retrieval, hybrid image query, RRF, web UI; three case studies.

---

## Slide 5: System Overview and CPU-only Configuration
Figure: `fig2_architecture.png`

| Component | Choice |
|---|---|
| Shots & keyframes | TransNetV2, FFmpeg |
| Visual encoder | OpenCLIP ViT-H/14 (DFN-5B) |
| Speech (optional) | Whisper Large-v3, EmbeddingGemma |
| Translation | Gemini 3.5 Flash-Lite |

- FAISS index: embedding retrieval, one search per query.
- Qdrant: metadata (video, frame ID, timestamp, path) and filtering.

---

## Slide 6: Offline Indexing and In-memory Retrieval
Figure: `fig1_indexing.png`

- **Keyframes:** four per shot at a + floor(i(b-a)/3), i in {0,1,2,3}. Timestamp t = frame index / FPS.
- **Embeddings:** unit vectors e_j = f(x_j) / ||f(x_j)||_2 are indexed in FAISS for retrieval. Image tower f and text tower g share one space.
- **Metadata:** JSON in Qdrant: video name, frame ID, timestamp, file path.

Key message: Retrieval runs on the in-memory FAISS index; Qdrant stores the keyframe metadata.

---

## Slide 7: Text and Speech Retrieval
**Query pipeline (flow):** Text query (Vietnamese / English) -> Translate & shorten (Gemini 3.5 Flash-Lite) -> Encode with g(.) to a unit vector q -> Search FAISS: s_j = q^T e_j -> Top-K, K = 50; info from Qdrant

**Speech (ASR) branch:** Whisper Large-v3 transcribes the audio. Each segment is linked to the keyframe nearest to its midpoint and embedded by EmbeddingGemma-300M.

**Reciprocal Rank Fusion:** when a speech query is given, visual and ASR rankings are merged:
RRF(d) = sum over modalities m of 1 / (c + r_m(d)), c = 60, where r_m(d) is the rank of keyframe d in modality m.

---

## Slide 8: Hybrid Image Query
Figure: `fig3_image_query.png`

Exact match first, similarity as fallback:
- **Offline:** worker threads compute the SHA-256 digest of every keyframe file into a hash index.
- **Hit:** return the source keyframe with score 1.0; the encoder is not run.
- **Miss:** z = f(image), then search the FAISS index (cosine, s_j = z^T e_j) for the top-K.
- **OOK concepts:** optional example images from Google Image Search, encoded with f.

Key message: An exact match is returned with certainty; a similar frame is ranked by similarity.

---

## Slide 9: TRAKE: Temporal Retrieval and Alignment of Key Events
Figure: `fig4_trake_workflow.png`

Two changes to the recurrence of Luu et al.:
- Gaps are counted in seconds, so the result does not depend on keyframe density.
- The context is mixed into every event embedding, so it takes no token budget.

Key message: The alignment needs no LLM call; FAISS retrieves embeddings, Qdrant supplies keyframe info.

---

## Slide 10: TRAKE: Context-aware Event Representation
**Vector blending:** Gemini 3.5 Flash-Lite translates the event and context texts and checks their length. With q_l = g(y_l) and u = g(y_0):
q_hat_l = ((1-w) q_l + w u) / ||(1-w) q_l + w u||_2, w = 0.3

**Why blend vectors, not text?**
- Appending the context to each event text could exceed the token limit.
- Since 1 - w = 0.7, the event vector still dominates; the context only nudges it.
- w, lambda and T_max are set once for all queries.

Key message: Example: events "a scene featuring a man" and "... a man and a woman" with context "at a news studio" (Case 2).

---

## Slide 11: TRAKE: Temporal Alignment via Dynamic Programming
**Objective** (score s_l(p) = q_hat_l^T e_p):
max over p_1..p_L of sum_l s_l(p_l) - lambda * sum_l (t_{p_{l+1}} - t_{p_l}), lambda = 0.0008 per second.
The costs add up to lambda (t_{p_L} - t_{p_1}): short chains are preferred.

**Recurrence:**
D_l(p) = s_l(p) - lambda t_p + max over p' in C_{l-1}, t_{p'} < t_p of [D_{l-1}(p') + lambda t_{p'}]
D_1(p) = s_1(p)

**Implementation**
- Every keyframe of a video is a candidate C_l for every event.
- Running maximum in time order; a binary search finds the last earlier keyframe.
- Cost O(L n log n) per video, L <= 4.
- T_max = 300 s is a filter after alignment, not part of the recurrence.
- Linear-penalty formulation of Luu et al. (2025); no new algorithm is claimed.

---

## Slide 12: Web Interface
Figure: `fig5_ui.png`

- **A:** Query, ASR, TRAKE context and events, reference image.
- **B:** Up to four selected frames of one video: KIS (1 frame) or TRAKE (2-4 frames).
- **C:** Search, TRAKE, Look up image, Reset, DRES login.
- **D:** Thumbnails with ID, score, timestamp; hover: IR, Video, Submit, Neighbors.

Key message: FastAPI web interface in Vietnamese and English; answers go to the DRES evaluation server.

---

## Slide 13: Case 1: Temporal Ordering
Figures: `fig9a_case1_top1.png` (a) top-1 frame of each event searched alone; `fig9b_case1_trake.png` (b) TRAKE sequence: swimming -> basketball -> piano.

Key message: Top-1 frames lie in three videos. TRAKE returns L25_V088 at 18:19-18:23 (4 s), trading 0.11 of summed score for the required order in one video.

---

## Slide 14: Case 2: Effect of the Shared Context
Figures: `fig10c_case2_noctx.png` (c) without context: the two top sequences; `fig10d_case2_ctx.png` (d) with the context "at a news studio", w = 0.3.

Key message: The context biases the retrieved sequences toward the intended scene without adding text tokens. Scores with and without context are not comparable.

---

## Slide 15: Case 3: Four Events in a Cluttered Video
Figure: `fig11_case3.png`

**Query and result:** Lantern-festival report: red star, female presenter, man in a dark gray shirt, toy panda. TRAKE returns one sequence from L22_V010 at 06:04, 06:16, 06:28, 06:29 (25 s); scores 0.4427, 0.3693, 0.2754, 0.4241.

**Effect of lambda:** lambda = 0 returns the same sequence; the penalty is only 0.0008 x 25 = 0.02. A wider chain wins only if its extra similarity exceeds lambda x Delta: 0.08 for 100 s, 0.24 for 300 s. No empirical gain from lambda is claimed.

---

## Slide 16: Results in the Final Round and CPU Cost
**Ratings, AIC 2026 final round**

| Task | Search mode | Rating |
|---|---|---|
| T-KIS | Semantic search (text) | Very good |
| V-KIS | Image lookup | Very good |
| TRAKE | Temporal alignment | Good |

(Note: Table 2 in the paper lists different ratings; verify against the official results.)

**CPU-only cost (median of 50 queries):** Intel Core i5-7200U @ 2.50 GHz, N = 384,989 keyframes
- Encode + score a query: 762 ms
- TRAKE alignment: 4380 ms

**How to read these results**
- Ratings are the only retrieval-quality measure; they are coarse and per task.
- They allow no comparison with other systems and no attribution to components.
- TRAKE received the lowest of the three ratings, although the core method was designed for it.
- w, lambda and T_max were not ablated; per-event candidate quality may also limit TRAKE.

---

## Slide 17: Conclusion and Future Directions
**Contributions**
1. In-memory FAISS retrieval.
2. TRAKE: DP over timestamps + context blending.
3. Hybrid image query and RRF fusion.
4. Web interface and three case studies.

**Future work**
- Ablate w, lambda, T_max.
- Enforce T_max inside the recurrence.
- Compare with other temporal methods.

**Findings and limitations**
- (+) Very good in T-KIS and V-KIS.
- (~) Good in TRAKE, the lowest of three.
- (x) No ablation; cases are single queries.
- (x) Translation needs an external API.
- (x) Memory grows with keyframes.

Key message: A CPU-only system can serve interactive multi-event queries; the evidence remains limited.

---

## Slide 18: THE END
Thank you for your attention.

Acknowledgements: We thank HCMUT - VNU-HCM, the HCMUT EE Machine Learning & IoT Lab and the Watt's Up Debuggers Club for their support of this research.

Contact: huy.tranminh2005@hcmut.edu.vn, nguyetminh261986@gmail.com, nkloi@hcmut.edu.vn

---

# Backup slides

## B1: Complete Formulation
- t = frame index / FPS; e_j = f(x_j) / ||f(x_j)||_2; s_j = z^T e_j
- RRF(d) = sum_m 1 / (c + r_m(d)), c = 60
- q_hat_l = ((1-w) q_l + w u) / ||(1-w) q_l + w u||_2, w = 0.3
- max sum_l s_l(p_l) - lambda sum_l (t_{p_{l+1}} - t_{p_l}), s_l(p) = q_hat_l^T e_p, lambda = 0.0008
- D_l(p) = s_l(p) - lambda t_p + max over p' in C_{l-1}, t_{p'} < t_p of [D_{l-1}(p') + lambda t_{p'}]; D_1(p) = s_1(p)
- Constraints: nu(p_1) = ... = nu(p_L), t_{p_1} < ... < t_{p_L}; chains longer than T_max = 300 s are dropped after alignment.

## B2: Parameters and Configuration
| Retrieval and TRAKE | Value |
|---|---|
| Keyframes per shot | 4 |
| Collection size N | 384,989 |
| Top-K (text / image) | 50 |
| RRF constant c | 60 |
| Context weight w | 0.3 |
| Gap penalty lambda | 0.0008 per second |
| Span limit T_max | 300 s |
| Events per query L | 1 to 4 |

| Models and tools | Choice |
|---|---|
| Shot detection | TransNetV2 |
| Frame rate | FFmpeg |
| Visual embeddings | OpenCLIP ViT-H/14 (DFN-5B) |
| Speech | Whisper Large-v3 |
| Text embeddings | EmbeddingGemma-300M |
| Translation | Gemini 3.5 Flash-Lite |
| Embedding index | FAISS |
| Keyframe metadata | Qdrant |
| Server | FastAPI, DRES |

## B3: Related Work and Positioning
| Approach | Reference | Handling of ordered events |
|---|---|---|
| Fixed window | Vo et al. (2025) | Window around a pivot frame |
| Index-based DP | Luu et al. (2025) | Dynamic program over keyframe indices |
| Anchor-based | Vu et al. (2025) | Beam search over anchors and an LLM |
| TRAKE | this work | Linear-penalty DP (after Luu et al.) with gaps in seconds, a candidate set per event, context blended into embeddings, no LLM |

Many systems query a vector database (e.g. Qdrant) for every request; we retrieve with an in-memory FAISS index and use Qdrant only for keyframe metadata.

## B4: Semantic Search and Image Lookup Examples
Figures: `fig6_semantic.png` (semantic search, translation enabled: costume festival in an African country); `fig7_imagelookup.png` (image lookup: banana photo; no hash match, 50 closest frames).
Single queries, shown as illustrations only. Scores are comparable only within one result list; image-to-image and text-to-image scores are on different scales.

## B5: TRAKE Example: Fruit Orchard
Figure: `fig8_trake_example.png`
Context: fruit orchard; events: durian, mangosteen, pomelo, bon bon fruit. Top sequence from L27_V011: durian, mangosteen, pomelo at 02:33, 02:36, 02:41; the fourth keyframe (02:47) has the lowest score (0.2885) and shows people walking in the orchard.

## B6: Case 1: Per-event Scores
| Event | Top-1 alone (video, time, score) | TRAKE (video, time, score) |
|---|---|---|
| Person swimming | L25_V070, 17:51, 0.3568 | L25_V088, 18:19, 0.3507 |
| Four people playing basketball | L28_V008, 17:46, 0.3449 | L25_V088, 18:20, 0.3271 |
| Person playing the piano | L25_V001, 01:40, 0.4149 | L25_V088, 18:23, 0.3250 |
| Sum of scores | 1.1166 | 1.0028 |

None of the top-5 results of any event comes from L25_V088. Each TRAKE frame scores below the fifth-ranked frame of its event list (0.3533, 0.3295, 0.4113).
