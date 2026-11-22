import React, { useState, useRef } from "react"
import {
  Upload,
  Sliders,
  AlertTriangle,
  FileText,
  FileSpreadsheet,
  RefreshCw,
  Sparkles,
  ShieldCheck,
  ChevronRight
} from "lucide-react"

interface AnalysisResult {
  status: string
  correlation_id: string
  filename: string
  timestamp: string
  dimensions: { width: number; height: number }
  overall_health_index: number
  restoration_priority: string
  urgency_phrase: string
  total_damaged_area_pct: number
  channel_contributions: {
    cracks: number
    moisture: number
    peeling: number
    biological: number
    erosion: number
  }
  priority_counts: {
    CRITICAL: number
    HIGH: number
    MODERATE: number
    LOW: number
  }
  detected_regions_count: number
  regions: Array<{
    region_id: number
    primary_damage: string
    primary_damage_name: string
    priority: string
    severity_score: number
    area_pct: number
    bbox: [number, number, number, number]
    is_compound: boolean
    channel_overlaps?: string[]
    ml_inference?: {
      predicted_class: string
      confidence: number
      calibrated_confidence: number
      ensemble_support: Record<string, number>
    }
    diagnostic_explanation: string
    recommendations: string[]
  }>
  visualizations?: {
    annotated_overlay_base64?: string
    heatmap_blended_base64?: string
  }
}

const SAMPLE_MONUMENTS = [
  {
    id: "konark",
    title: "Konark Sun Temple",
    location: "Odisha, 13th Century",
    type: "Kalinga Sandstone & Khondalite",
    image: "/images/konark_sun_temple.jpg",
    filename: "konark_sun_temple.jpg",
  },
  {
    id: "gopuram",
    title: "Golden Dravidian Gopuram",
    location: "South India, Dravidian Order",
    type: "Granite Core & Sculpted Stucco",
    image: "/images/golden_gopuram.jpg",
    filename: "golden_gopuram.jpg",
  },
  {
    id: "stepwell",
    title: "Carved Subterranean Stepwell",
    location: "Gujarat / Rajasthan Vav",
    type: "Deep Subsurface Sandstone",
    image: "/images/carved_stepwell.jpg",
    filename: "carved_stepwell.jpg",
  },
  {
    id: "haveli",
    title: "Ornate Sandstone Jharokha",
    location: "Jaisalmer / Jodhpur Haveli",
    type: "Fretted Yellow Sandstone",
    image: "/images/haveli_sandstone.jpg",
    filename: "haveli_sandstone.jpg",
  },
]

export const AnalysisDashboard: React.FC = () => {
  // State
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string>("/images/haveli_sandstone.jpg")
  const [activeSampleId, setActiveSampleId] = useState<string>("haveli")
  
  // Detection Settings
  const [sensitivity, setSensitivity] = useState<number>(0.55)
  const [detectCracks, setDetectCracks] = useState<boolean>(true)
  const [detectMoisture, setDetectMoisture] = useState<boolean>(true)
  const [detectPeeling, setDetectPeeling] = useState<boolean>(true)
  const [detectBio, setDetectBio] = useState<boolean>(true)
  const [detectErosion, setDetectErosion] = useState<boolean>(true)
  const [applyClahe, setApplyClahe] = useState<boolean>(true)
  const [colormap, setColormap] = useState<string>("JET")
  
  // Execution & Results
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false)
  const [isExportingPdf, setIsExportingPdf] = useState<boolean>(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [activeTab, setActiveTab] = useState<"overlay" | "heatmap" | "original">("overlay")
  const [expandedRegionId, setExpandedRegionId] = useState<number | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  // Load sample image
  const handleSelectSample = async (sample: typeof SAMPLE_MONUMENTS[0]) => {
    setActiveSampleId(sample.id)
    setPreviewUrl(sample.image)
    try {
      const response = await fetch(sample.image)
      const blob = await response.blob()
      const file = new File([blob], sample.filename, { type: "image/jpeg" })
      setSelectedImage(file)
      setResult(null)
      setErrorMsg(null)
    } catch (err) {
      console.error("Failed to load sample image blob", err)
    }
  }

  // Handle file upload
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setSelectedImage(file)
      setPreviewUrl(URL.createObjectURL(file))
      setActiveSampleId("custom")
      setResult(null)
      setErrorMsg(null)
    }
  }

  // Execute Analysis via API
  const runAnalysis = async () => {
    setIsAnalyzing(true)
    setErrorMsg(null)

    try {
      let fileToUpload = selectedImage
      if (!fileToUpload) {
        // Fallback fetch active preview URL if user clicked analyze without uploading
        const response = await fetch(previewUrl)
        const blob = await response.blob()
        fileToUpload = new File([blob], "monument_wall.jpg", { type: "image/jpeg" })
        setSelectedImage(fileToUpload)
      }

      const formData = new FormData()
      formData.append("file", fileToUpload)

      const queryParams = new URLSearchParams({
        sensitivity: sensitivity.toString(),
        enable_cracks: detectCracks.toString(),
        enable_moisture: detectMoisture.toString(),
        enable_peeling: detectPeeling.toString(),
        enable_biological: detectBio.toString(),
        enable_erosion: detectErosion.toString(),
        apply_clahe: applyClahe.toString(),
        colormap: colormap,
        return_images: "true",
      })

      // In production with Vite proxy or Cloudflare Pages, use relative or env-configured URL
      const apiUrl = `/api/v1/analyze?${queryParams.toString()}`

      const res = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || `Server responded with status ${res.status}`)
      }

      const data: AnalysisResult = await res.json()
      setResult(data)
    } catch (err: any) {
      console.error("Analysis request failed", err)
      setErrorMsg(err.message || "Failed to analyze image. Please ensure the backend server is running on port 8000.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Export PDF Report
  const handleExportPdf = async () => {
    if (!selectedImage && !previewUrl) return
    setIsExportingPdf(true)
    try {
      let fileToUpload = selectedImage
      if (!fileToUpload) {
        const response = await fetch(previewUrl)
        const blob = await response.blob()
        fileToUpload = new File([blob], "monument_wall.jpg", { type: "image/jpeg" })
      }

      const formData = new FormData()
      formData.append("file", fileToUpload)

      const res = await fetch("/api/v1/report/pdf", {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        throw new Error("Failed to generate PDF report from server.")
      }

      const pdfBlob = await res.blob()
      const url = window.URL.createObjectURL(pdfBlob)
      const a = document.createElement("a")
      a.href = url
      a.download = `KalaRaksha_Conservation_Report_${Date.now()}.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (err: any) {
      alert("Error generating PDF: " + (err.message || "Unknown error"))
    } finally {
      setIsExportingPdf(false)
    }
  }

  // Export CSV
  const handleExportCsv = () => {
    if (!result || !result.regions) return
    const headers = [
      "Region ID",
      "Primary Damage",
      "Priority",
      "Severity Score (0-1)",
      "Area Coverage %",
      "ML Predicted Class",
      "ML Confidence %",
      "Compound Deterioration",
      "Primary Recommendation",
    ]

    const rows = result.regions.map((r) => [
      r.region_id,
      `"${r.primary_damage_name}"`,
      r.priority,
      r.severity_score.toFixed(3),
      r.area_pct.toFixed(2),
      r.ml_inference?.predicted_class || "N/A",
      r.ml_inference?.confidence ? (r.ml_inference.confidence * 100).toFixed(1) : "N/A",
      r.is_compound ? "Yes" : "No",
      `"${(r.recommendations?.[0] || "").replace(/"/g, '""')}"`,
    ])

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n")

    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `KalaRaksha_Damage_Inventory_${Date.now()}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Priority color styling helper
  const getPriorityBadgeClass = (priority: string) => {
    switch (priority.toUpperCase()) {
      case "CRITICAL":
        return "bg-red-500/10 text-red-400 border-red-500/30"
      case "HIGH":
        return "bg-orange-500/10 text-orange-400 border-orange-500/30"
      case "MODERATE":
        return "bg-yellow-500/10 text-yellow-400 border-yellow-500/30"
      case "LOW":
      default:
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
    }
  }

  return (
    <div id="workbench" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12 text-left">
      
      {/* Section Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-6 border-b border-zinc-800">
        <div>
          <span className="text-xs font-mono font-bold tracking-widest text-amber-400 uppercase">
            Interactive Diagnostic Workbench
          </span>
          <h2 className="text-3xl font-extrabold text-white mt-1 font-serif">
            Heritage Wall Deterioration Analysis
          </h2>
          <p className="text-zinc-400 text-sm mt-1 max-w-2xl">
            Select an authentic monument or upload field imagery to trigger calibrated multi-hazard edge, chromatic, and morphological decomposition.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={runAnalysis}
            disabled={isAnalyzing}
            className="px-6 py-3 rounded-xl font-semibold text-zinc-950 bg-gradient-to-r from-amber-400 to-yellow-500 hover:from-amber-300 hover:to-yellow-400 shadow-lg shadow-amber-500/20 disabled:opacity-50 transition-all flex items-center gap-2 cursor-pointer"
          >
            {isAnalyzing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-zinc-950" />
                <span>Decomposing Layers...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 text-zinc-950" />
                <span>Run IPA + ML Analysis</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Grid: Left Config & Uploader / Right Visualizer & Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: Sample Selector, Uploader & Detector Controls */}
        <div className="lg:col-span-4 space-y-6">
          
          {/* Sample Selector */}
          <div className="bg-zinc-900/60 rounded-2xl p-5 border border-zinc-800">
            <label className="text-xs font-mono font-bold text-zinc-400 uppercase tracking-wider block mb-3">
              1. Authentic Heritage Benchmarks
            </label>
            <div className="grid grid-cols-2 gap-2.5">
              {SAMPLE_MONUMENTS.map((m) => (
                <button
                  key={m.id}
                  onClick={() => handleSelectSample(m)}
                  className={`text-left p-2.5 rounded-xl border transition-all text-xs cursor-pointer flex flex-col justify-between ${
                    activeSampleId === m.id
                      ? "bg-amber-950/40 border-amber-500/80 shadow-md shadow-amber-900/20"
                      : "bg-zinc-950/40 border-zinc-800 hover:border-zinc-700"
                  }`}
                >
                  <div className="h-16 w-full rounded-lg overflow-hidden mb-2 bg-zinc-800">
                    <img
                      src={m.image}
                      alt={m.title}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <span className="font-semibold text-zinc-200 line-clamp-1">{m.title}</span>
                  <span className="text-[10px] text-zinc-500 line-clamp-1 mt-0.5">{m.location}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Custom File Uploader */}
          <div className="bg-zinc-900/60 rounded-2xl p-5 border border-zinc-800">
            <label className="text-xs font-mono font-bold text-zinc-400 uppercase tracking-wider block mb-3">
              2. Or Upload Custom Field Photo
            </label>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
            />
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-zinc-700 hover:border-amber-500/60 rounded-xl p-6 text-center cursor-pointer transition-colors bg-zinc-950/30 group"
            >
              <Upload className="w-8 h-8 text-zinc-500 group-hover:text-amber-400 mx-auto mb-2 transition-colors" />
              <p className="text-xs font-medium text-zinc-300">
                Click or drag & drop high-res image
              </p>
              <p className="text-[10px] text-zinc-500 mt-1">
                JPG, PNG, WEBP (Max 25MB · Auto-resized to 1280px)
              </p>
            </div>
          </div>

          {/* IPA Pipeline Settings */}
          <div className="bg-zinc-900/60 rounded-2xl p-5 border border-zinc-800 space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-mono font-bold text-zinc-400 uppercase tracking-wider">
                3. Detector Configuration
              </label>
              <Sliders className="w-3.5 h-3.5 text-amber-400" />
            </div>

            {/* Sensitivity Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-zinc-400">Sensitivity Threshold</span>
                <span className="text-amber-400 font-mono font-semibold">
                  {(sensitivity * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0.10"
                max="1.00"
                step="0.05"
                value={sensitivity}
                onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                className="w-full accent-amber-400 bg-zinc-800 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* Detector Toggles */}
            <div className="space-y-2 pt-2 text-xs">
              {[
                { label: "Structural & Hairline Cracks", state: detectCracks, setter: setDetectCracks, color: "text-red-400" },
                { label: "Moisture Stains & Damp Seepage", state: detectMoisture, setter: setDetectMoisture, color: "text-blue-400" },
                { label: "Peeling Paint & Detached Plaster", state: detectPeeling, setter: setDetectPeeling, color: "text-orange-400" },
                { label: "Biological Growth (Moss / Algae)", state: detectBio, setter: setDetectBio, color: "text-emerald-400" },
                { label: "Surface Erosion & Grain Roughness", state: detectErosion, setter: setDetectErosion, color: "text-amber-400" },
              ].map((det, i) => (
                <label key={i} className="flex items-center justify-between p-2 rounded-lg bg-zinc-950/40 border border-zinc-800/80 cursor-pointer hover:bg-zinc-800/30">
                  <span className={`font-medium ${det.color}`}>{det.label}</span>
                  <input
                    type="checkbox"
                    checked={det.state}
                    onChange={(e) => det.setter(e.target.checked)}
                    className="accent-amber-500 rounded cursor-pointer"
                  />
                </label>
              ))}
            </div>

            {/* CLAHE & Colormap */}
            <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
              <div>
                <span className="text-zinc-500 block mb-1">Color Palette</span>
                <select
                  value={colormap}
                  onChange={(e) => setColormap(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-zinc-300 font-mono text-xs focus:outline-none focus:border-amber-500"
                >
                  <option value="JET">Thermal JET</option>
                  <option value="TURBO">Google TURBO</option>
                  <option value="INFERNO">Perceptual INFERNO</option>
                </select>
              </div>

              <div>
                <span className="text-zinc-500 block mb-1">Contrast Normalization</span>
                <label className="flex items-center gap-2 p-2 rounded-lg bg-zinc-950 border border-zinc-800 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={applyClahe}
                    onChange={(e) => setApplyClahe(e.target.checked)}
                    className="accent-amber-500 rounded"
                  />
                  <span className="text-zinc-300">LAB CLAHE</span>
                </label>
              </div>
            </div>

          </div>

        </div>

        {/* Right Column: Visualizer & Results */}
        <div className="lg:col-span-8 space-y-6">
          
          {/* Main Visual Display Card */}
          <div className="bg-zinc-900/60 rounded-2xl p-5 border border-zinc-800 shadow-xl overflow-hidden">
            
            {/* View Switcher Tabs */}
            <div className="flex items-center justify-between pb-4 border-b border-zinc-800 mb-4">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActiveTab("overlay")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-colors ${
                    activeTab === "overlay"
                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  Multi-Hazard Overlay
                </button>
                <button
                  onClick={() => setActiveTab("heatmap")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-colors ${
                    activeTab === "heatmap"
                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  Deterioration Heatmap
                </button>
                <button
                  onClick={() => setActiveTab("original")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-colors ${
                    activeTab === "original"
                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  Original Image
                </button>
              </div>

              {result && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleExportPdf}
                    disabled={isExportingPdf}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium text-zinc-300 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                  >
                    <FileText className="w-3.5 h-3.5 text-amber-400" />
                    <span>{isExportingPdf ? "Generating..." : "PDF Dossier"}</span>
                  </button>
                  <button
                    onClick={handleExportCsv}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium text-zinc-300 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 flex items-center gap-1.5 cursor-pointer"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
                    <span>CSV Data</span>
                  </button>
                </div>
              )}
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div className="p-4 mb-4 rounded-xl bg-red-950/40 border border-red-800/60 text-red-300 text-xs flex items-center gap-3">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Image Stage */}
            <div className="relative aspect-[4/3] rounded-xl overflow-hidden bg-zinc-950 border border-zinc-800 flex items-center justify-center">
              {isAnalyzing && (
                <div className="absolute inset-0 z-20 bg-black/70 backdrop-blur-sm flex flex-col items-center justify-center space-y-3">
                  <RefreshCw className="w-8 h-8 text-amber-400 animate-spin" />
                  <p className="text-sm font-medium text-amber-200">
                    Running Morphological & Chromatic Decomposition...
                  </p>
                  <p className="text-xs text-zinc-400">
                    Extracting crack geometry, moisture gradient & random forest features
                  </p>
                </div>
              )}

              {activeTab === "overlay" && result?.visualizations?.annotated_overlay_base64 ? (
                <img
                  src={result.visualizations.annotated_overlay_base64}
                  alt="Multi-Hazard Annotated Overlay"
                  className="w-full h-full object-contain"
                />
              ) : activeTab === "heatmap" && result?.visualizations?.heatmap_blended_base64 ? (
                <img
                  src={result.visualizations.heatmap_blended_base64}
                  alt="Deterioration Thermal Heatmap"
                  className="w-full h-full object-contain"
                />
              ) : (
                <img
                  src={previewUrl}
                  alt="Source Monument Wall"
                  className="w-full h-full object-contain"
                />
              )}

              {/* Status Badge */}
              <div className="absolute bottom-3 left-3 z-10">
                <span className="px-2.5 py-1 rounded-md text-[11px] font-mono font-medium tracking-wide bg-black/80 backdrop-blur-md text-zinc-300 border border-zinc-700">
                  {result
                    ? `Resolved: ${result.dimensions.width}×${result.dimensions.height}px · ${result.detected_regions_count} Zones`
                    : "Ready for Analysis"}
                </span>
              </div>
            </div>

            {/* Overview Metric Banners (Rendered upon analysis) */}
            {result && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800">
                  <span className="text-[10px] font-mono text-zinc-500 uppercase block">Restoration Priority</span>
                  <span className={`inline-block mt-1 px-2 py-0.5 rounded text-xs font-bold border ${getPriorityBadgeClass(result.restoration_priority)}`}>
                    {result.restoration_priority}
                  </span>
                  <p className="text-[11px] text-zinc-400 mt-1 line-clamp-1">{result.urgency_phrase}</p>
                </div>

                <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800">
                  <span className="text-[10px] font-mono text-zinc-500 uppercase block">Health Index</span>
                  <span className="text-xl font-bold font-mono text-amber-400 mt-0.5 block">
                    {result.overall_health_index.toFixed(1)}%
                  </span>
                  <p className="text-[11px] text-zinc-400 mt-0.5">Structural Integrity</p>
                </div>

                <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800">
                  <span className="text-[10px] font-mono text-zinc-500 uppercase block">Damaged Surface</span>
                  <span className="text-xl font-bold font-mono text-orange-400 mt-0.5 block">
                    {result.total_damaged_area_pct.toFixed(2)}%
                  </span>
                  <p className="text-[11px] text-zinc-400 mt-0.5">Aggregated Coverage</p>
                </div>

                <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800">
                  <span className="text-[10px] font-mono text-zinc-500 uppercase block">Critical Zones</span>
                  <span className="text-xl font-bold font-mono text-red-400 mt-0.5 block">
                    {result.priority_counts.CRITICAL} Zones
                  </span>
                  <p className="text-[11px] text-zinc-400 mt-0.5">Immediate Intervention</p>
                </div>
              </div>
            )}

          </div>

          {/* Damage Channels Distribution */}
          {result && (
            <div className="bg-zinc-900/60 rounded-2xl p-5 border border-zinc-800">
              <h3 className="text-sm font-mono font-bold text-zinc-300 uppercase tracking-wider mb-4">
                Channel Decomposition Ratios
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                {[
                  { name: "Cracks", pct: result.channel_contributions.cracks, color: "bg-red-500" },
                  { name: "Moisture", pct: result.channel_contributions.moisture, color: "bg-blue-500" },
                  { name: "Peeling", pct: result.channel_contributions.peeling, color: "bg-orange-500" },
                  { name: "Biological", pct: result.channel_contributions.biological, color: "bg-emerald-500" },
                  { name: "Erosion", pct: result.channel_contributions.erosion, color: "bg-amber-500" },
                ].map((item, idx) => (
                  <div key={idx} className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800/80">
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span className="text-zinc-400">{item.name}</span>
                      <span className="font-mono font-semibold text-zinc-200">{item.pct.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${item.color} rounded-full`}
                        style={{ width: `${Math.min(100, item.pct * 2)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Quantitative Damage Regions Table */}
          {result && result.regions && result.regions.length > 0 && (
            <div className="bg-zinc-900/60 rounded-2xl p-5 border border-zinc-800 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-mono font-bold text-zinc-300 uppercase tracking-wider">
                    Localized Damage Region Catalog ({result.regions.length})
                  </h3>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    Individual connected components scored against calibrated ICOMOS / ASI preservation thresholds
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-zinc-300">
                  <thead className="text-[11px] uppercase tracking-wider font-mono text-zinc-500 bg-zinc-950/80 border-b border-zinc-800">
                    <tr>
                      <th className="py-2.5 px-3">Zone ID</th>
                      <th className="py-2.5 px-3">Hazard Category</th>
                      <th className="py-2.5 px-3">Area %</th>
                      <th className="py-2.5 px-3">Severity</th>
                      <th className="py-2.5 px-3">Priority</th>
                      <th className="py-2.5 px-3">ML Diagnostic</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/60 font-sans">
                    {result.regions.map((reg) => (
                      <React.Fragment key={reg.region_id}>
                        <tr className="hover:bg-zinc-800/30 transition-colors">
                          <td className="py-3 px-3 font-mono font-bold text-amber-400">
                            #{reg.region_id}
                          </td>
                          <td className="py-3 px-3">
                            <span className="font-semibold text-zinc-200">
                              {reg.primary_damage_name}
                            </span>
                            {reg.is_compound && (
                              <span className="ml-1.5 px-1.5 py-0.5 rounded text-[9px] font-mono bg-purple-500/10 text-purple-400 border border-purple-500/30">
                                COMPOUND
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-3 font-mono text-zinc-300">
                            {reg.area_pct.toFixed(2)}%
                          </td>
                          <td className="py-3 px-3 font-mono">
                            <span
                              className={
                                reg.severity_score >= 0.7
                                  ? "text-red-400 font-semibold"
                                  : reg.severity_score >= 0.4
                                  ? "text-orange-400"
                                  : "text-zinc-400"
                              }
                            >
                              {(reg.severity_score * 10).toFixed(1)}/10
                            </span>
                          </td>
                          <td className="py-3 px-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${getPriorityBadgeClass(reg.priority)}`}>
                              {reg.priority}
                            </span>
                          </td>
                          <td className="py-3 px-3">
                            {reg.ml_inference ? (
                              <span className="inline-flex items-center gap-1 text-[11px] text-zinc-400 font-mono">
                                <ShieldCheck className="w-3 h-3 text-emerald-400" />
                                {reg.ml_inference.predicted_class} ({(reg.ml_inference.confidence * 100).toFixed(0)}%)
                              </span>
                            ) : (
                              <span className="text-zinc-600 font-mono">IPA Only</span>
                            )}
                          </td>
                          <td className="py-3 px-3 text-right">
                            <button
                              onClick={() =>
                                setExpandedRegionId(
                                  expandedRegionId === reg.region_id ? null : reg.region_id
                                )
                              }
                              className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-mono text-[10px] cursor-pointer"
                            >
                              {expandedRegionId === reg.region_id ? "Close" : "Details"}
                            </button>
                          </td>
                        </tr>

                        {/* Collapsible Detail Row */}
                        {expandedRegionId === reg.region_id && (
                          <tr className="bg-zinc-950/70">
                            <td colSpan={7} className="p-4 space-y-3">
                              <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800">
                                <span className="text-[10px] font-mono text-amber-400 uppercase tracking-wider block mb-1">
                                  Diagnostic Explanation
                                </span>
                                <p className="text-xs text-zinc-300 leading-relaxed">
                                  {reg.diagnostic_explanation}
                                </p>
                              </div>

                              {reg.recommendations && reg.recommendations.length > 0 && (
                                <div className="p-3 rounded-xl bg-amber-950/20 border border-amber-800/30">
                                  <span className="text-[10px] font-mono text-amber-300 uppercase tracking-wider block mb-1.5">
                                    Preservation Directives (ASI/ICOMOS)
                                  </span>
                                  <ul className="space-y-1 text-xs text-zinc-300">
                                    {reg.recommendations.map((rec, rIdx) => (
                                      <li key={rIdx} className="flex items-start gap-2">
                                        <ChevronRight className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                                        <span>{rec}</span>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </div>

      </div>

    </div>
  )
}
