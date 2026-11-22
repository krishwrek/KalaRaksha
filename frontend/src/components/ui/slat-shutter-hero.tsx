import React, { useState } from "react"
import { motion } from "motion/react"
import { Sparkles, Eye, ShieldAlert, ArrowRight, RotateCcw } from "lucide-react"
import { cn } from "@/lib/utils"

export interface SlatShutterHeroProps {
  badgeText?: string
  title?: string
  subtitle?: string
  description?: string
  coverSrc: string
  revealSrc: string
  slatCount?: number
  primaryAction?: {
    label: string
    href?: string
    onClick?: () => void
  }
  secondaryAction?: {
    label: string
    href?: string
    onClick?: () => void
  }
  className?: string
}

export const SlatShutterHero: React.FC<SlatShutterHeroProps> = ({
  badgeText = "KALARAKSHA · HERITAGE RESTORATION INTELLIGENCE",
  title = "Preserving the Sacred Architecture of Time",
  subtitle = "Explainable Computer Vision & Diagnostic ML for Monumental Wall Conservation",
  description = "Analyze micro-cracks, deep moisture seepage, plaster detachment, biological crusts, and mechanical abrasion across ancient temple sanctums, murals, and stepwells with calibrated archaeological rigor.",
  coverSrc,
  revealSrc,
  slatCount = 10,
  primaryAction,
  secondaryAction,
  className,
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const slats = Array.from({ length: slatCount }, (_, i) => i)

  return (
    <section className={cn("relative overflow-hidden bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 py-16 lg:py-24 border-b border-amber-950/40", className)}>
      {/* Decorative Heritage Radial Glows */}
      <div className="absolute top-0 left-1/4 -translate-y-1/2 w-96 h-96 bg-amber-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute top-1/3 right-10 w-80 h-80 bg-orange-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          
          {/* Left Column: Typography & CTAs */}
          <div className="lg:col-span-6 space-y-6 text-left">
            {badgeText && (
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider bg-amber-500/10 border border-amber-500/30 text-amber-300 shadow-inner"
              >
                <Sparkles className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
                <span>{badgeText}</span>
              </motion.div>
            )}

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-tight font-serif"
            >
              {title.split(" ").map((word, idx) => (
                <span
                  key={idx}
                  className={
                    idx === 1 || idx === 2
                      ? "text-transparent bg-clip-text bg-gradient-to-r from-amber-400 via-orange-400 to-amber-200"
                      : ""
                  }
                >
                  {word}{" "}
                </span>
              ))}
            </motion.h1>

            {subtitle && (
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="text-lg sm:text-xl font-medium text-amber-200/90"
              >
                {subtitle}
              </motion.p>
            )}

            {description && (
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="text-base text-zinc-400 leading-relaxed max-w-xl"
              >
                {description}
              </motion.p>
            )}

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="flex flex-wrap items-center gap-4 pt-2"
            >
              {primaryAction && (
                <button
                  onClick={primaryAction.onClick}
                  className="px-6 py-3.5 rounded-xl font-semibold text-zinc-950 bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-500 hover:from-amber-300 hover:to-yellow-400 shadow-lg shadow-amber-500/20 hover:shadow-amber-500/30 transition-all transform hover:-translate-y-0.5 active:translate-y-0 flex items-center gap-2 cursor-pointer"
                >
                  <span>{primaryAction.label}</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              )}

              {secondaryAction && (
                <button
                  onClick={secondaryAction.onClick}
                  className="px-6 py-3.5 rounded-xl font-semibold text-zinc-300 bg-zinc-900/80 hover:bg-zinc-800 border border-zinc-700/80 hover:border-zinc-600 transition-all flex items-center gap-2 cursor-pointer"
                >
                  <ShieldAlert className="w-4 h-4 text-amber-400" />
                  <span>{secondaryAction.label}</span>
                </button>
              )}

              {/* Shutter Toggle Controller */}
              <button
                onClick={() => setIsOpen(!isOpen)}
                className="px-4 py-3 rounded-xl text-xs font-semibold text-amber-300/90 bg-amber-950/30 border border-amber-800/40 hover:bg-amber-950/60 transition-all flex items-center gap-2 cursor-pointer ml-auto sm:ml-0"
                title="Toggle slat shutter inspection"
              >
                {isOpen ? <RotateCcw className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                <span>{isOpen ? "Close Shutter" : "Reveal Architecture (Slat View)"}</span>
              </button>
            </motion.div>

            {/* Quick Metrics Strip */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.5 }}
              className="grid grid-cols-3 gap-4 pt-4 border-t border-zinc-800/80 text-zinc-400"
            >
              <div>
                <p className="text-xl font-bold text-amber-400 font-mono">5 IPA</p>
                <p className="text-xs text-zinc-500">Detectors (Crack, Moisture, Peeling, Bio, Roughness)</p>
              </div>
              <div>
                <p className="text-xl font-bold text-emerald-400 font-mono">Calibrated ML</p>
                <p className="text-xs text-zinc-500">Platt-scaled RandomForest Confidence</p>
              </div>
              <div>
                <p className="text-xl font-bold text-orange-400 font-mono">ICOMOS / ASI</p>
                <p className="text-xs text-zinc-500">Standardized Restoration Priority Protocol</p>
              </div>
            </motion.div>
          </div>

          {/* Right Column: Slat Shutter Visualizer */}
          <div className="lg:col-span-6">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8 }}
              className="relative mx-auto w-full max-w-lg aspect-[4/5] rounded-2xl overflow-hidden border-2 border-amber-600/30 shadow-2xl shadow-black/80 group"
              onMouseEnter={() => setIsOpen(true)}
              onMouseLeave={() => setIsOpen(false)}
            >
              {/* Underlying Reveal Image (Layer 0) */}
              <div
                className="absolute inset-0 bg-cover bg-center transition-transform duration-700 group-hover:scale-105"
                style={{ backgroundImage: `url(${revealSrc})` }}
              >
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                <div className="absolute bottom-4 left-4 right-4 p-3 bg-zinc-950/80 backdrop-blur-md rounded-xl border border-amber-500/30 text-left">
                  <span className="text-[10px] uppercase font-mono font-bold tracking-widest text-amber-400 block">
                    Underlying Monolithic Profile
                  </span>
                  <p className="text-xs text-zinc-200 font-medium">
                    Golden Dravidian Gopuram · Multi-Tiered Granite & Stucco Relief
                  </p>
                </div>
              </div>

              {/* Slat Shutter Slices (Layer 1 - Cover Image) */}
              <div className="absolute inset-0 flex pointer-events-none">
                {slats.map((index) => {
                  const widthPercent = 100 / slatCount
                  const backgroundPositionX = `${(index / (slatCount - 1)) * 100}%`

                  return (
                    <motion.div
                      key={index}
                      className="h-full relative overflow-hidden"
                      style={{
                        width: `${widthPercent}%`,
                      }}
                      animate={{
                        scaleY: isOpen ? 0.08 : 1,
                        opacity: isOpen ? 0.25 : 1,
                        rotateX: isOpen ? 45 : 0,
                      }}
                      transition={{
                        type: "spring",
                        stiffness: 120,
                        damping: 18,
                        delay: index * 0.03,
                      }}
                    >
                      <div
                        className="absolute inset-0 h-full w-[500px] max-w-none"
                        style={{
                          backgroundImage: `url(${coverSrc})`,
                          backgroundSize: "cover",
                          backgroundPosition: `${backgroundPositionX} center`,
                          left: `-${index * (500 / slatCount)}px`,
                        }}
                      />
                      <div className="absolute inset-0 bg-black/10 shadow-[inset_0_0_10px_rgba(0,0,0,0.5)] border-r border-black/30" />
                    </motion.div>
                  )
                })}
              </div>

              {/* Hover Cue Overlay */}
              <div className="absolute top-4 right-4 z-20 pointer-events-none">
                <span className="px-2.5 py-1 rounded-md text-[11px] font-mono font-medium tracking-wide bg-black/70 backdrop-blur-md text-amber-300 border border-amber-500/30 shadow-lg">
                  {isOpen ? "SHUTTER OPEN" : "HOVER TO SHUTTER"}
                </span>
              </div>
            </motion.div>

            <p className="text-center text-xs text-zinc-500 mt-3 font-mono">
              Hover over facade or tap button to operate the kinetic slat shutters
            </p>
          </div>

        </div>
      </div>
    </section>
  )
}
