"use client"

import { useEffect, useState } from "react"
import { Play, Pause, Trash2, Activity, Target, Clock } from "lucide-react"

interface Detection {
  id: string
  timestamp: number
  filename: string
  confidence: number
  bbox: {
    x: number
    y: number
    width: number
    height: number
    x1: number
    y1: number
    x2: number
    y2: number
  }
  image_original: string
  image_annotated: string
  receivedAt: string
}

export default function Dashboard() {
  const [detections, setDetections] = useState<Detection[]>([])
  const [isPolling, setIsPolling] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Polling des détections
  useEffect(() => {
    if (!isPolling) return

    const fetchDetections = async () => {
      try {
        const res = await fetch("/api/detections")
        if (!res.ok) throw new Error("Erreur réseau")

        const data = await res.json()
        setDetections(data.detections)
        setError(null)
      } catch (err) {
        setError("Impossible de charger les détections")
        console.error("Erreur polling:", err)
      }
    }

    fetchDetections()
    const interval = setInterval(fetchDetections, 500)

    return () => clearInterval(interval)
  }, [isPolling])

  // Calculer statistiques
  const avgConfidence =
    detections.length > 0 ? (detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length).toFixed(1) : 0

  const clearDetections = async () => {
    if (!confirm("Effacer toutes les détections ?")) return

    try {
      await fetch("/api/detections", { method: "DELETE" })
      setDetections([])
    } catch (err) {
      console.error("Erreur suppression:", err)
    }
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
              Détection de Drones FPV
            </h1>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">Monitoring en temps réel</p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPolling(!isPolling)}
              className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                isPolling
                  ? "bg-zinc-900 text-zinc-50 hover:bg-zinc-800 dark:bg-zinc-50 dark:text-zinc-900 dark:hover:bg-zinc-200"
                  : "bg-zinc-200 text-zinc-900 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-50 dark:hover:bg-zinc-700"
              }`}
            >
              {isPolling ? (
                <>
                  <Pause className="h-4 w-4" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Resume
                </>
              )}
            </button>

            <button
              onClick={clearDetections}
              className="inline-flex items-center gap-2 rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-900 transition-colors hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50 dark:hover:bg-zinc-800"
            >
              <Trash2 className="h-4 w-4" />
              Clear
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="mb-8 grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-3">
              <div className="rounded-full bg-zinc-100 p-2 dark:bg-zinc-800">
                <Target className="h-5 w-5 text-zinc-600 dark:text-zinc-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Total Détections</p>
                <p className="mt-1 text-2xl font-semibold text-zinc-900 dark:text-zinc-50">{detections.length}</p>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-3">
              <div className="rounded-full bg-zinc-100 p-2 dark:bg-zinc-800">
                <Activity className="h-5 w-5 text-zinc-600 dark:text-zinc-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Confiance Moyenne</p>
                <p className="mt-1 text-2xl font-semibold text-zinc-900 dark:text-zinc-50">{avgConfidence}%</p>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-3">
              <div className="rounded-full bg-zinc-100 p-2 dark:bg-zinc-800">
                <Clock className="h-5 w-5 text-zinc-600 dark:text-zinc-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Dernière Détection</p>
                <p className="mt-1 text-xl font-semibold text-zinc-900 dark:text-zinc-50">
                  {detections.length > 0 ? new Date(detections[0].receivedAt).toLocaleTimeString("fr-FR") : "-"}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* État polling */}
        <div className="mb-6 flex items-center gap-2 text-sm">
          <div className={`h-2 w-2 rounded-full ${isPolling ? "bg-emerald-500 animate-pulse" : "bg-zinc-400"}`} />
          <span className="text-zinc-600 dark:text-zinc-400">{isPolling ? "Écoute active" : "En pause"}</span>
          {error && <span className="ml-4 text-red-600 dark:text-red-400">{error}</span>}
        </div>

        {/* Grille des détections */}
        {detections.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-300 bg-white py-20 text-center dark:border-zinc-700 dark:bg-zinc-900">
            <div className="mx-auto max-w-sm">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-zinc-100 dark:bg-zinc-800">
                <Target className="h-6 w-6 text-zinc-400" />
              </div>
              <h3 className="mb-1 text-lg font-medium text-zinc-900 dark:text-zinc-50">En attente de détections</h3>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">Les détections apparaîtront ici en temps réel</p>
            </div>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {detections.map((detection) => (
              <div
                key={detection.id}
                className="group overflow-hidden rounded-lg border border-zinc-200 bg-white transition-shadow hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900"
              >
                {/* Image */}
                <div className="relative aspect-video bg-zinc-100 dark:bg-zinc-800">
                  <img
                    src={`data:image/jpeg;base64,${detection.image_annotated}`}
                    alt={detection.filename}
                    className="h-full w-full object-contain"
                  />

                  {/* Badge confiance */}
                  <div className="absolute right-3 top-3 rounded-full bg-emerald-500 px-2.5 py-1 text-xs font-semibold text-white">
                    {detection.confidence}%
                  </div>

                  {/* Badge temps */}
                  <div className="absolute bottom-3 left-3 rounded-full bg-zinc-900/80 px-2.5 py-1 text-xs font-medium text-zinc-100 backdrop-blur-sm">
                    {new Date(detection.receivedAt).toLocaleTimeString("fr-FR")}
                  </div>
                </div>

                {/* Specs */}
                <div className="p-4">
                  <h3
                    className="mb-3 truncate text-sm font-medium text-zinc-900 dark:text-zinc-50"
                    title={detection.filename}
                  >
                    {detection.filename}
                  </h3>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="rounded border border-zinc-200 bg-zinc-50 p-2 dark:border-zinc-800 dark:bg-zinc-800/50">
                      <span className="mb-1 block font-medium text-zinc-500 dark:text-zinc-400">Position Centre</span>
                      <div className="font-mono text-zinc-900 dark:text-zinc-50">
                        ({detection.bbox.x}, {detection.bbox.y})
                      </div>
                    </div>

                    <div className="rounded border border-zinc-200 bg-zinc-50 p-2 dark:border-zinc-800 dark:bg-zinc-800/50">
                      <span className="mb-1 block font-medium text-zinc-500 dark:text-zinc-400">Taille</span>
                      <div className="font-mono text-zinc-900 dark:text-zinc-50">
                        {detection.bbox.width}×{detection.bbox.height}
                      </div>
                    </div>

                    <div className="rounded border border-zinc-200 bg-zinc-50 p-2 dark:border-zinc-800 dark:bg-zinc-800/50">
                      <span className="mb-1 block font-medium text-zinc-500 dark:text-zinc-400">BBox X</span>
                      <div className="font-mono text-zinc-900 dark:text-zinc-50">
                        [{detection.bbox.x1}, {detection.bbox.x2}]
                      </div>
                    </div>

                    <div className="rounded border border-zinc-200 bg-zinc-50 p-2 dark:border-zinc-800 dark:bg-zinc-800/50">
                      <span className="mb-1 block font-medium text-zinc-500 dark:text-zinc-400">BBox Y</span>
                      <div className="font-mono text-zinc-900 dark:text-zinc-50">
                        [{detection.bbox.y1}, {detection.bbox.y2}]
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
