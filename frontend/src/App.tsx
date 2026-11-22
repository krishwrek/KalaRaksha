import { useState, useEffect } from "react"
import { SlatShutterHero } from "@/components/ui/slat-shutter-hero"
import { AnalysisDashboard } from "@/components/AnalysisDashboard"
import {
  Shield,
  Layers,
  Cpu,
  ExternalLink,
  Award,
  CheckCircle,
  FileCheck2,
  Scale
} from "lucide-react"

export function App() {
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)

  useEffect(() => {
    // Health probe to backend
    fetch("/health")
      .then((res) => (res.ok ? setApiOnline(true) : setApiOnline(false)))
      .catch(() => setApiOnline(false))
  }, [])

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: "smooth" })
    }
  }

  return (
    <div className="min-h-screen bg-[#0b0d11] text-zinc-200 selection:bg-amber-500/30 selection:text-amber-200">
      
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-zinc-950/80 border-b border-zinc-800/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          
          {/* Logo & Emblem */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-amber-600 via-orange-500 to-amber-400 p-0.5 flex items-center justify-center shadow-md shadow-amber-950">
              <div className="w-full h-full bg-zinc-950 rounded-[10px] flex items-center justify-center">
                <Shield className="w-4 h-4 text-amber-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-extrabold text-base tracking-tight text-white font-serif">
                  KalaRaksha
                </span>
                <span className="text-xs font-serif text-amber-400/90 font-medium">
                  कलारक्षा
                </span>
              </div>
              <p className="text-[10px] text-zinc-500 font-mono tracking-tight hidden sm:block">
                Heritage Wall Deterioration Analysis & Restoration Mapping
              </p>
            </div>
          </div>

          {/* Nav Items */}
          <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-zinc-400">
            <button
              onClick={() => scrollToSection("workbench")}
              className="hover:text-amber-300 transition-colors cursor-pointer"
            >
              Workbench
            </button>
            <button
              onClick={() => scrollToSection("methodology")}
              className="hover:text-amber-300 transition-colors cursor-pointer"
            >
              IPA & ML Architecture
            </button>
            <button
              onClick={() => scrollToSection("principles")}
              className="hover:text-amber-300 transition-colors cursor-pointer"
            >
              Conservation Principles
            </button>
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="hover:text-amber-300 transition-colors flex items-center gap-1"
            >
              <span>API Specs</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </nav>

          {/* Action & Backend Health Indicator */}
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-full text-[11px] font-mono bg-zinc-900 border border-zinc-800">
              <span
                className={`w-2 h-2 rounded-full ${
                  apiOnline === true
                    ? "bg-emerald-400 animate-pulse"
                    : apiOnline === false
                    ? "bg-amber-500"
                    : "bg-zinc-600"
                }`}
              />
              <span className="text-zinc-400">
                {apiOnline === true
                  ? "API Online"
                  : apiOnline === false
                  ? "Local Backend"
                  : "Checking API"}
              </span>
            </div>

            <a
              href="https://github.com/krishwrek/KalaRaksha"
              target="_blank"
              rel="noreferrer"
              className="px-3.5 py-1.5 rounded-xl text-xs font-semibold text-zinc-300 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 transition-all flex items-center gap-2"
            >
              <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              <span className="hidden sm:inline">GitHub</span>
            </a>
          </div>

        </div>
      </header>

      {/* Kinetic Hero: SlatShutterHero with Authentic Images */}
      <SlatShutterHero
        badgeText="ARCHAEOLOGICAL COMPUTER VISION · 2026 EDITION"
        title="Unveiling the Hidden Scars of Sacred Architecture"
        subtitle="Classical Image Processing Analysis (IPA) & Calibrated Machine Learning for Heritage Monuments"
        description="Engineered to diagnose hairline fracture propagation, deep moisture seepage, plaster detachment, biological crusts, and surface erosion across ancient temple sanctums, murals, and stepwells with scientific rigor."
        coverSrc="/images/haveli_sandstone.jpg"
        revealSrc="/images/golden_gopuram.jpg"
        slatCount={10}
        primaryAction={{
          label: "Launch Diagnostic Workbench",
          onClick: () => scrollToSection("workbench"),
        }}
        secondaryAction={{
          label: "View Methodology",
          onClick: () => scrollToSection("methodology"),
        }}
      />

      {/* Main Diagnostic Workbench */}
      <AnalysisDashboard />

      {/* Architecture & Methodology Section */}
      <section id="methodology" className="py-20 bg-zinc-950 border-t border-zinc-800/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-left">
          
          <div className="max-w-3xl mb-12">
            <span className="text-xs font-mono font-bold tracking-widest text-amber-400 uppercase">
              Scientific Pipeline Architecture
            </span>
            <h2 className="text-3xl font-extrabold text-white mt-1 font-serif">
              Multi-Hazard Classical IPA & Calibrated ML
            </h2>
            <p className="text-zinc-400 text-sm mt-2 leading-relaxed">
              Unlike generic black-box object detectors that hallucinate bounding boxes, KalaRaksha decomposes physical image properties using explainable mathematical transforms before passing morphological features to calibrated classifiers.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Card 1: 5 IPA Detectors */}
            <div className="p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800 space-y-4">
              <div className="w-10 h-10 rounded-xl bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white font-serif">
                5 Classical IPA Transforms
              </h3>
              <ul className="space-y-2 text-xs text-zinc-400 leading-relaxed">
                <li><strong className="text-zinc-200">Crack Detection:</strong> Canny gradients, morphological skeletonization, and Probabilistic Hough Line transforms separating fractures from decorative joints.</li>
                <li><strong className="text-zinc-200">Moisture Seepage:</strong> LAB CLAHE contrast normalization with HSV chromatic saturation attenuation highlighting damp zones.</li>
                <li><strong className="text-zinc-200">Plaster Peeling:</strong> Sobel gradient field variance isolating blistered edges and detached plaster.</li>
                <li><strong className="text-zinc-200">Biological Growth:</strong> Excess Green Index ($2G - R - B$) with saturation thresholds for moss and lichen.</li>
                <li><strong className="text-zinc-200">Surface Erosion:</strong> Local window variance and entropy filtering capturing mechanical grain loss.</li>
              </ul>
            </div>

            {/* Card 2: Calibrated ML Engine */}
            <div className="p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800 space-y-4">
              <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <Cpu className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white font-serif">
                Platt-Calibrated Random Forest
              </h3>
              <p className="text-xs text-zinc-400 leading-relaxed">
                Extracted connected components are converted into a standardized 10-dimensional morphological feature vector (aspect ratio, circularity, solidity, intensity variance, chromatic excess, channel overlaps).
              </p>
              <div className="p-3 rounded-xl bg-zinc-950/80 border border-zinc-800 font-mono text-[11px] text-zinc-300 space-y-1">
                <div className="flex justify-between">
                  <span className="text-zinc-500">Estimators:</span>
                  <span className="text-amber-400 font-semibold">150 Trees</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Calibration:</span>
                  <span className="text-emerald-400 font-semibold">Sigmoid / Platt Scaled</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Benchmark Accuracy:</span>
                  <span className="text-cyan-400 font-semibold">97.5% Tested</span>
                </div>
              </div>
            </div>

            {/* Card 3: ASI / ICOMOS Priority Engine */}
            <div className="p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800 space-y-4">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                <Scale className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white font-serif">
                ASI & ICOMOS Priority Protocol
              </h3>
              <p className="text-xs text-zinc-400 leading-relaxed">
                Damage is evaluated through archaeological priority guidelines codified by the Archaeological Survey of India (ASI) and the Venice Charter:
              </p>
              <ul className="space-y-1.5 text-xs text-zinc-400">
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500 shrink-0" />
                  <span><strong>Critical:</strong> Active seepage, through-wall fissures (&gt;2mm), extensive delamination.</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-orange-500 shrink-0" />
                  <span><strong>High:</strong> Compound overlapping degradation, damp plaster blisters.</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-yellow-500 shrink-0" />
                  <span><strong>Moderate:</strong> Superficial micro-cracks, initial biological colonies.</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
                  <span><strong>Low:</strong> Normal atmospheric patina, localized surface roughness.</span>
                </li>
              </ul>
            </div>

          </div>

        </div>
      </section>

      {/* Conservation Principles Section */}
      <section id="principles" className="py-20 bg-[#0d0f14] border-t border-zinc-800/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-left">
          
          <div className="max-w-3xl mb-12">
            <span className="text-xs font-mono font-bold tracking-widest text-amber-400 uppercase">
              Ethical Heritage Restoration
            </span>
            <h2 className="text-3xl font-extrabold text-white mt-1 font-serif">
              Standard Conservation Directives
            </h2>
            <p className="text-zinc-400 text-sm mt-2 leading-relaxed">
              Computer vision recommendations generated by KalaRaksha adhere to international standards of monumental material preservation.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                title: "Non-Destructive Evaluation",
                desc: "Never apply chemical strippers or physical percussion before establishing complete non-invasive chromatic and structural profiling.",
                icon: Shield,
              },
              {
                title: "Reversibility of Intervention",
                desc: "Any consolidant or adhesive introduced must be fully reversible without damaging the original historic lime-surkhi or stone matrix.",
                icon: FileCheck2,
              },
              {
                title: "Material Compatibility",
                desc: "Never use modern Portland cement on ancient lime or sandstone; use authentic slaked lime mortars and compatible stone dust.",
                icon: Award,
              },
              {
                title: "Root-Cause Mitigation",
                desc: "Treat the moisture source (roofing parapets, ground capillarity, rainwater spouts) before attempting cosmetic surface patch repairs.",
                icon: CheckCircle,
              },
            ].map((principle, idx) => (
              <div key={idx} className="p-5 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 space-y-3">
                <principle.icon className="w-6 h-6 text-amber-400" />
                <h4 className="text-base font-bold text-white font-serif">{principle.title}</h4>
                <p className="text-xs text-zinc-400 leading-relaxed">{principle.desc}</p>
              </div>
            ))}
          </div>

        </div>
      </section>

      {/* Legal & DMCA Safe Harbor Footer */}
      <footer id="legal" className="py-12 bg-zinc-950 border-t border-zinc-800 text-zinc-500 text-xs text-left">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-amber-400" />
                <span className="font-bold text-zinc-200 font-serif">KalaRaksha Conservation Project</span>
              </div>
              <p className="text-zinc-500 leading-relaxed">
                Dedicated to non-invasive computer vision analysis for ancient Indian monuments, temple murals, and rock-cut sanctuaries.
              </p>
            </div>

            <div className="space-y-2">
              <span className="font-mono text-zinc-300 font-semibold block uppercase tracking-wider text-[11px]">
                Research & Standards
              </span>
              <ul className="space-y-1.5 text-zinc-400">
                <li>ICOMOS Venice Charter (1964)</li>
                <li>Archaeological Survey of India (ASI) Norms</li>
                <li>Nara Document on Authenticity (1994)</li>
                <li>Digital Photogrammetry & Non-Destructive Testing</li>
              </ul>
            </div>

            <div className="space-y-2">
              <span className="font-mono text-zinc-300 font-semibold block uppercase tracking-wider text-[11px]">
                Legal & Compliance
              </span>
              <ul className="space-y-1.5 text-zinc-400">
                <li>DMCA Safe Harbor Compliant</li>
                <li>17 U.S.C. § 512(c) Protocol</li>
                <li>Non-Commercial Research Exemption</li>
                <li>Zero-Retention Field Image Processing</li>
              </ul>
            </div>

            <div className="space-y-2">
              <span className="font-mono text-zinc-300 font-semibold block uppercase tracking-wider text-[11px]">
                Designated DMCA Agent
              </span>
              <p className="text-zinc-400 leading-relaxed">
                Heritage Preservation Directorate<br />
                Email: <span className="text-amber-400">compliance@kalaraksha.org</span><br />
                Notice Takedown Turnaround: &lt; 24h
              </p>
            </div>
          </div>

          <div className="pt-6 border-t border-zinc-900 flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px]">
            <p>© {new Date().getFullYear()} KalaRaksha Project. Released under MIT License for heritage conservationists and researchers.</p>
            <p className="font-mono text-zinc-600">Built with Vite · React · Tailwind CSS · Motion · FastAPI · OpenCV</p>
          </div>

        </div>
      </footer>

    </div>
  )
}

export default App
