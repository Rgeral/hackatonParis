"use client"

import { useEffect, useState } from "react"
import { Play, Pause, Trash2, Activity, Target, Clock, Map, Grid3x3 } from "lucide-react"
import dynamic from "next/dynamic"

const DetectionMap = dynamic(() => import("@/components/DetectionMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[600px] flex items-center justify-center bg-zinc-900 rounded-lg border-2 border-zinc-700">
      Chargement de la carte...
    </div>
  ),
})

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
  camera_id?: string
  camera_location?: { lat: number; lng: number; name: string }
  image_dimensions?: { width: number; height: number }
  image_original: string
  image_annotated: string
  receivedAt: string
}

export default function Dashboard() {
  const [detections, setDetections] = useState<Detection[]>([])
  const [isPolling, setIsPolling] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<"grid" | "map">("grid")
  const [alerts, setAlerts] = useState<Array<{ id: string; message: string; level: string; timestamp: number }>>([])

  // Fonction pour gérer les alertes de danger
  const handleDangerAlert = (message: string, level: string) => {
    const newAlert = {
      id: `alert_${Date.now()}_${Math.random()}`,
      message,
      level,
      timestamp: Date.now(),
    }
    
    setAlerts((prev) => [newAlert, ...prev.slice(0, 4)]) // Garder max 5 alertes
    
    // Notification navigateur si supporté et autorisé
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification(level === "critical" ? "🚨 ALERTE CRITIQUE" : "⚠️ ALERTE DANGER", {
        body: message,
        icon: "/favicon.ico",
        tag: newAlert.id,
      })
    }
    
    // Auto-dismiss après 10 secondes
    setTimeout(() => {
      setAlerts((prev) => prev.filter((a) => a.id !== newAlert.id))
    }, 10000)
  }

  // Demander la permission pour les notifications au chargement
  useEffect(() => {
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission()
    }
  }, [])

  // Polling des détections
  useEffect(() => {
    if (!isPolling) return

    const fetchDetections = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_DETECTIONS_URL || "/api/detections"
        const res = await fetch(apiUrl)
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
    const pollingInterval = parseInt(process.env.NEXT_PUBLIC_POLLING_INTERVAL || "2000", 10)
    const interval = setInterval(fetchDetections, pollingInterval)

    return () => clearInterval(interval)
  }, [isPolling])

  // Calculer statistiques
  const avgConfidence =
    detections.length > 0 ? (detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length).toFixed(1) : 0

  const clearDetections = async () => {
    if (!confirm("Effacer toutes les détections ?")) return

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_DETECTIONS_URL || "/api/detections"
      await fetch(apiUrl, { method: "DELETE" })
      setDetections([])
    } catch (err) {
      console.error("Erreur suppression:", err)
    }
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      {/* Système d'alertes en haut à droite */}
      <div className="fixed right-4 top-4 z-50 space-y-2 max-w-md">
        {alerts.map((alert) => (
          <div
            key={alert.id}
            className={`rounded-lg border-2 p-4 shadow-lg backdrop-blur-sm animate-in slide-in-from-right ${
              alert.level === "critical"
                ? "border-red-600 bg-red-50/95 dark:bg-red-950/95"
                : "border-orange-500 bg-orange-50/95 dark:bg-orange-950/95"
            }`}
          >
            <div className="flex items-start gap-3">
              <div className="text-2xl">
                {alert.level === "critical" ? "🚨" : "⚠️"}
              </div>
              <div className="flex-1">
                <p
                  className={`text-sm font-medium leading-relaxed ${
                    alert.level === "critical"
                      ? "text-red-900 dark:text-red-100"
                      : "text-orange-900 dark:text-orange-100"
                  }`}
                >
                  {alert.message}
                </p>
                <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
                  {new Date(alert.timestamp).toLocaleTimeString("fr-FR")}
                </p>
              </div>
              <button
                onClick={() => setAlerts((prev) => prev.filter((a) => a.id !== alert.id))}
                className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200"
              >
                ✕
              </button>
            </div>
          </div>
        ))}
      </div>

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

        {/* Onglets */}
        <div className="mb-6 flex gap-2 border-b border-zinc-200 dark:border-zinc-800">
          <button
            onClick={() => setActiveTab("grid")}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-[1px] ${
              activeTab === "grid"
                ? "border-zinc-900 text-zinc-900 dark:border-zinc-50 dark:text-zinc-50"
                : "border-transparent text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-50"
            }`}
          >
            <Grid3x3 className="h-4 w-4" />
            Grille
          </button>
          <button
            onClick={() => setActiveTab("map")}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-[1px] ${
              activeTab === "map"
                ? "border-zinc-900 text-zinc-900 dark:border-zinc-50 dark:text-zinc-50"
                : "border-transparent text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-50"
            }`}
          >
            <Map className="h-4 w-4" />
            Carte
          </button>
        </div>

        {/* Contenu basé sur l'onglet actif */}
        {activeTab === "map" ? (
          <div className="rounded-lg">
            <DetectionMap detections={detections} onDangerAlert={handleDangerAlert} />
          </div>
        ) : detections.length === 0 ? (
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
