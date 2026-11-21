# KalaRaksha: Algorithmic Architecture & Mathematical Foundations

## Overview
**KalaRaksha** is an explainable digital conservation and Image Processing Analysis (IPA) system engineered specifically for historical monuments, temple walls, frescoes, murals, bas-reliefs, and stone facades. 

Unlike opaque deep-learning models that require heavy GPUs and produce arbitrary bounding boxes without scientific justification, KalaRaksha employs **deterministic, physically-grounded image processing algorithms**. Every pixel flagged as deteriorated is tied to concrete mathematical metrics (e.g. gradient energy, chromaticity indices, spatial entropy, or geometric aspect ratios).

---

## 1. Image Preprocessing Pipeline (`utils/preprocessing.py`)

### 1.1 Resolution Bounding
Input images of any size are constrained such that:
$$\max(H, W) \le 1280 \text{ pixels}$$
$$\text{Aspect Ratio} = \frac{W_{\text{orig}}}{H_{\text{orig}}} \equiv \text{constant}$$
This ensures high spatial fidelity for hairline cracks while allowing deterministic real-time processing on standard commodity CPUs without GPU acceleration.

### 1.2 Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB Color Space
Direct equalization in RGB introduces artificial color cast. Instead, the image is transformed to the CIE $L^*a^*b^*$ color space, where Lightness ($L$) is decoupled from chromaticity channels ($a, b$):
$$L_{\text{enhanced}} = \text{CLAHE}(L, \text{clipLimit}=2.5, \text{tileGridSize}=(8, 8))$$
CLAHE divides the image into $8 \times 8$ contextual tiles. Histogram bins exceeding the clip limit are redistributed uniformly before bilinear interpolation across tile boundaries, amplifying micro-fractures in deep shadows.

### 1.3 Edge-Preserving Denoising
Bilateral filtering smooths high-frequency sensor noise while preserving sharp crack boundaries and flaking paint ridges:
$$I^{\text{filtered}}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(\|I(x_i) - I(x)\|) g_s(\|x_i - x\|)$$
where $g_s$ is the spatial Gaussian kernel and $f_r$ is the radiometric range Gaussian kernel.

---

## 2. Crack Detection Engine (`utils/crack_detection.py`)

1. **Morphological Black-Hat Transform:**
   Isolates dark linear fissures on lighter stone or plaster backgrounds:
   $$T_{\text{blackhat}}(I) = \text{Closing}(I, K) - I = (I \bullet K) - I$$
   using multi-scale elliptical structuring elements ($7 \times 7$ and $15 \times 15$).

2. **Adaptive Dynamic Canny Thresholding:**
   Anchored around median intensity $v_{\text{med}}$:
   $$\sigma = 0.35 \times (1.6 - 0.7 \times \text{sensitivity})$$
   $$T_{\text{lower}} = \max(15, (1 - \sigma) \cdot v_{\text{med}}), \quad T_{\text{upper}} = \min(240, (1 + \sigma) \cdot v_{\text{med}})$$

3. **Probabilistic Hough Line Transform:**
   Extracts linear fissure propagation vectors:
   $$R = x \cos \theta + y \sin \theta$$

4. **Morphological & Geometric Filtering:**
   Cracks are filtered from decorative motifs or mortar lines using:
   - **Elongation / Aspect Ratio:** $\frac{\max(w, h)}{\min(w, h)} \ge 2.5$
   - **Circularity / Compactness:** $C = \frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2} \le 0.35$
   - **Thickness Estimation:** $T_{\text{est}} = \frac{\text{Area}}{\text{Perimeter} / 2} \le 18 \text{ px}$

5. **1-Pixel Centerline Skeletonization:**
   Thinning extracts the topological skeleton to compute exact physical fissure length:
   $$L_{\text{total}} = \sum \text{NonZero}(\text{Skeleton})$$

---

## 3. Moisture Stains & Seepage Detection (`utils/moisture_detection.py`)

Water ingress darkens porous stone due to refractive index changes and deposits mineral salts:
1. **Background Illumination Estimation:**
   Local luminance $L$ is compared against a large-neighborhood Gaussian illumination baseline:
   $$L_{\text{diff}} = G_{\sigma}(L) - L, \quad \sigma \approx \frac{\min(H, W)}{15}$$
2. **Chromatic Dampness Verification:**
   Damp watermarks and rising damp mineral stains cluster in brownish/dark HSV bands:
   $$H \in [8^\circ, 38^\circ], \quad S \ge 20, \quad V \le 180$$
3. **Amoeboid Contour Validation:**
   Contiguous plumes are validated via low compactness ($C \le 0.65$), distinguishing irregular rising damp plumes from rectangular architectural blocks.

---

## 4. Peeling Paint & Plaster Delamination (`utils/peeling_detection.py`)

1. **Sobel Gradient Magnitude:**
   Flaking edges create abrupt curling boundaries:
   $$G = \sqrt{G_x^2 + G_y^2} = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2}$$
2. **Local Edge-Density Convolution:**
   Flaking paint produces swarms of curling micro-edges rather than single isolated lines:
   $$D_{\text{edge}}(x, y) = \frac{1}{|W|} \sum_{(i,j) \in W} B_{\text{edge}}(x+i, y+j)$$
3. **Local Variance Step-Change:**
   Delaminated paint reveals rough exposed masonry underneath, producing a distinct jump in local standard deviation:
   $$\sigma_{\text{local}} = \sqrt{\mathbb{E}[I^2] - (\mathbb{E}[I])^2}$$

---

## 5. Biological Growth Detection (`utils/biological_growth.py`)

Identifies moss (bryophytes), green algae (chlorophyta), and crustose lichens:
1. **Excess Green Index (ExG):**
   Normalized RGB channels ($r, g, b = \frac{R, G, B}{R + G + B + \epsilon}$):
   $$\text{ExG} = 2g - r - b$$
   Biological chlorophyll deposits produce $\text{ExG} > 0.06$.
2. **HSV Chlorophyll Resonance:**
   $$H \in [26^\circ, 88^\circ], \quad S \ge 40, \quad V \ge 25$$
   Olive lichen: $H \in [18^\circ, 32^\circ], S \ge 45$.
3. **Green Channel Dominance Check:**
   Enforces $G_{\text{raw}} - R_{\text{raw}} > 8$ to distinguish biological green from pale pastel fresco paints.

---

## 6. Surface Erosion & Weathering (`utils/erosion_detection.py`)

Stone weathering causes granular disintegration and alveolar pitting (honeycomb decay):
1. **Local Standard Deviation ($\sigma$):** Captures high-frequency granular abrasion.
2. **Local Shannon Entropy ($H$):**
   $$H = -\sum_{i=0}^{255} p_i \log_2(p_i)$$
   Measures structural disorder in the stone relief.
3. **Laplacian Micro-Roughness:**
   $$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
4. **Fused Roughness Metric:**
   $$R_{\text{fused}} = 0.40 \cdot \sigma_{\text{norm}} + 0.35 \cdot H_{\text{norm}} + 0.25 \cdot |\nabla^2 I|_{\text{norm}}$$

---

## 7. Multi-Modal Heatmap Fusion (`utils/heatmap_fusion.py`)

The 5 normalized channels $D_k \in [0, 1]$ are combined via risk-weighted linear superposition:
$$H_{\text{raw}}(x, y) = \sum_{k=1}^5 w_k D_k(x, y), \quad \sum w_k = 1.0$$
- Cracks: $w = 0.30$ (Structural integrity threat)
- Moisture: $w = 0.25$ (Catalytic weathering agent)
- Peeling: $w = 0.15$ (Loss of decorative art layer)
- Biological: $w = 0.15$ (Biochemical stone dissolution)
- Erosion: $w = 0.15$ (Loss of surface relief and patina)

Thermal smoothing is applied via Gaussian convolution:
$$H_{\text{fused}} = H_{\text{raw}} * G_{\sigma=15}$$
Colored through false-color transfer functions (JET, TURBO, INFERNO) and blended with the enhanced wall image:
$$I_{\text{blend}} = (1 - \alpha) \cdot I_{\text{wall}} + \alpha \cdot \text{Colormap}(H_{\text{fused}})$$

---

## 8. Heritage Health Index & Priority Engine (`utils/priority_engine.py`)

### Heritage Health Index (HHI, 0 - 100)
$$HHI = 100 - \min\left(98, \text{Damage Penalty} + \sum \text{Priority Penalties}\right)$$
- $HHI \ge 80$: Good Standing / Routine Maintenance
- $60 \le HHI < 80$: Active Preventive Care Required
- $40 \le HHI < 60$: Urgent Conservation Scheduled
- $HHI < 40$: Immediate Emergency Intervention Required
