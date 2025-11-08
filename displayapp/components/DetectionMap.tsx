"use client"

import { useEffect, useRef } from "react"
import L from "leaflet"
import "leaflet/dist/leaflet.css"
import { evaluateDanger, estimateDirection, calculateBearingToUser } from "@/lib/droneEstimations"

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
  }
  camera_id?: string
  camera_location?: { lat: number; lng: number; name: string }
  image_dimensions?: { width: number; height: number }
  drone_heading?: number  // Cap du drone en degrés (optionnel)
  drone_location?: { lat: number; lng: number }  // Position GPS (optionnelle)
}

interface DetectionMapProps {
  detections: Detection[]
  onDangerAlert?: (message: string, level: string) => void
}

// Caméras mockées dans la région parisienne
const MOCK_CAMERAS = [
  { id: "CAM_01", lat: 48.8566, lng: 2.3522, name: "Tour Eiffel" },
  { id: "CAM_02", lat: 48.8738, lng: 2.295, name: "Arc de Triomphe" },
  { id: "CAM_03", lat: 48.8606, lng: 2.3376, name: "Louvre" },
  { id: "CAM_04", lat: 48.8529, lng: 2.3499, name: "Notre-Dame" },
]

// Position utilisateur mockée
const USER_POSITION = { lat: 48.8584, lng: 2.3448, name: "Vous êtes ici" }

// Gestion de la blacklist dans localStorage
const BLACKLIST_KEY = 'drone_detections_blacklist'

function getBlacklist(): Set<string> {
  if (typeof window === 'undefined') return new Set()
  try {
    const stored = localStorage.getItem(BLACKLIST_KEY)
    return stored ? new Set(JSON.parse(stored)) : new Set()
  } catch {
    return new Set()
  }
}

function addToBlacklist(detectionId: string) {
  const blacklist = getBlacklist()
  blacklist.add(detectionId)
  localStorage.setItem(BLACKLIST_KEY, JSON.stringify(Array.from(blacklist)))
}

export default function DetectionMap({ detections, onDangerAlert }: DetectionMapProps) {
  const mapRef = useRef<L.Map | null>(null)
  const mapContainerRef = useRef<HTMLDivElement>(null)
  const alertedDetectionsRef = useRef<Set<string>>(new Set())
  const droneLayersRef = useRef<Map<string, { circle: L.Circle; line: L.Polyline; arrow: L.Polyline; arrowHead: L.Circle }>>(new Map())
  const blacklistRef = useRef<Set<string>>(getBlacklist())

  // Initialiser la carte
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return

    // Nettoyer la Map des layers au montage (après un refresh)
    droneLayersRef.current.clear()

    // Créer la carte centrée sur Paris
    const map = L.map(mapContainerRef.current, {
      closePopupOnClick: false, // Empêcher la fermeture des popups au clic sur la carte
    }).setView([48.8566, 2.3522], 13)

    // Ajouter les tuiles OpenStreetMap sans labels (style CartoDB Positron No Labels)
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
      maxZoom: 19,
    }).addTo(map)

    // Ajouter les caméras
    MOCK_CAMERAS.forEach((camera) => {
      const cameraIcon = L.divIcon({
        html: `
          <div style="
            background-color: #3b82f6;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          ">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
          </div>
        `,
        className: "",
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      })

      L.marker([camera.lat, camera.lng], { icon: cameraIcon })
        .addTo(map)
        .bindTooltip(`<b>📷 ${camera.name}</b><br>ID: ${camera.id}`, {
          permanent: false,
          direction: "top",
          offset: [0, -16],
        })
        .bindPopup(`<b>${camera.name}</b><br>ID: ${camera.id}`)
    })

    // Ajouter la position utilisateur
    const userIcon = L.divIcon({
      html: `
        <div style="
          background-color: #10b981;
          width: 40px;
          height: 40px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          border: 4px solid white;
          box-shadow: 0 2px 12px rgba(16, 185, 129, 0.5);
        ">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
          </svg>
        </div>
      `,
      className: "",
      iconSize: [40, 40],
      iconAnchor: [20, 20],
    })

    L.marker([USER_POSITION.lat, USER_POSITION.lng], { icon: userIcon })
      .addTo(map)
      .bindTooltip(`<b>👤 ${USER_POSITION.name}</b>`, {
        permanent: false,
        direction: "top",
        offset: [0, -20],
      })
      .bindPopup(`<b>${USER_POSITION.name}</b>`)

    mapRef.current = map

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  // Mettre à jour les détections sur la carte
  useEffect(() => {
    console.log('🔄 UPDATE détections:', detections.length, 'détections reçues')
    
    if (!mapRef.current) {
      console.log('❌ Pas de map ref')
      return
    }

    const map = mapRef.current

    // Garder une trace des IDs actuels
    const currentDetectionIds = new Set(detections.slice(-20).map(d => d.id))
    console.log('📋 IDs actuels:', Array.from(currentDetectionIds))
    console.log('📋 IDs stockés:', Array.from(droneLayersRef.current.keys()))
    
    // Supprimer les layers des détections qui n'existent plus
    droneLayersRef.current.forEach((layers, detectionId) => {
      if (!currentDetectionIds.has(detectionId)) {
        console.log('🗑️ Suppression détection:', detectionId)
        map.removeLayer(layers.circle)
        map.removeLayer(layers.line)
        map.removeLayer(layers.arrow)
        map.removeLayer(layers.arrowHead)
        droneLayersRef.current.delete(detectionId)
      }
    })

    // Si pas de détections, on a fini (tout a été nettoyé ci-dessus)
    if (detections.length === 0) {
      console.log('⚠️ Aucune détection à afficher')
      return
    }

    // Ajouter les nouveaux drones
    detections.slice(-20).forEach((detection) => {
      // Ignorer les détections blacklistées
      if (blacklistRef.current.has(detection.id)) {
        console.log('🚫 Détection blacklistée, skip:', detection.id)
        return
      }
      
      // Si ce drone existe déjà, ne pas le recréer
      if (droneLayersRef.current.has(detection.id)) {
        console.log('⏭️ Détection déjà existante, skip:', detection.id)
        return
      }
      
      console.log('✨ Création nouvelle détection:', detection.id)
      
      const camera = MOCK_CAMERAS.find((cam) => cam.id === detection.camera_id) || MOCK_CAMERAS[0]

      // Estimation de distance basée sur bbox
      const bboxArea = detection.bbox.width * detection.bbox.height
      let distance = 500 // mètres par défaut

      if (bboxArea < 5000) distance = 500
      else if (bboxArea < 15000) distance = 300
      else if (bboxArea < 40000) distance = 150
      else distance = 75

      // Direction estimée en DEGRÉS (0-360°)
      const imgWidth = detection.image_dimensions?.width || 1920
      const imgHeight = detection.image_dimensions?.height || 1080

      const directionData = estimateDirection(detection.bbox, { width: imgWidth, height: imgHeight })
      const droneBearing = directionData.degrees
      const droneCardinal = directionData.cardinal

      // 🎯 POSITION DU DRONE : Utiliser drone_location si fourni, sinon calculer depuis bbox
      let droneLat: number
      let droneLng: number

      if (detection.drone_location) {
        // ✅ Position exacte fournie par le script
        droneLat = detection.drone_location.lat
        droneLng = detection.drone_location.lng
        console.log(`📍 Position drone fournie: ${droneLat.toFixed(6)}, ${droneLng.toFixed(6)}`)
      } else {
        // ⚠️ Calculer position approximative depuis bbox et caméra
        const metersPerDegree = 111000
        const offsetDegrees = distance / metersPerDegree

        // Convertir l'angle en composantes lat/lng
        const angleRad = (droneBearing * Math.PI) / 180
        const latOffset = Math.cos(angleRad) * offsetDegrees
        const lngOffset = Math.sin(angleRad) * offsetDegrees

        droneLat = camera.lat + latOffset
        droneLng = camera.lng + lngOffset
        console.log(`⚠️ Position drone calculée depuis bbox: ${droneLat.toFixed(6)}, ${droneLng.toFixed(6)}`)
      }

      // 🧭 CAP DU DRONE : Utiliser drone_heading si fourni, sinon calculer pire cas
      let droneHeading: number
      
      if (detection.drone_heading !== undefined) {
        // ✅ Cap fourni par le script Python
        droneHeading = detection.drone_heading
        console.log(`✈️ Cap drone fourni: ${droneHeading.toFixed(1)}°`)
      } else {
        // ⚠️ Pas de cap fourni, on suppose qu'il se dirige vers l'utilisateur (pire cas)
        const { degrees: bearingToUser } = calculateBearingToUser(
          { lat: droneLat, lng: droneLng },
          { lat: USER_POSITION.lat, lng: USER_POSITION.lng }
        )
        droneHeading = bearingToUser
        console.log(`⚠️ Cap drone non fourni, pire cas (vers utilisateur): ${droneHeading.toFixed(1)}°`)
      }

      // 📏 Calculer la vraie distance caméra→drone (GPS)
      const distanceCameraToDrone = Math.sqrt(
        Math.pow((droneLat - camera.lat) * 111000, 2) +
        Math.pow((droneLng - camera.lng) * 111000 * Math.cos(camera.lat * Math.PI / 180), 2)
      )
      
      console.log(`📏 Distance bbox (estimée): ${distance}m | Distance GPS réelle caméra→drone: ${distanceCameraToDrone.toFixed(1)}m`)

      // Utiliser la fonction d'évaluation de danger avec le cap du drone
      const dangerAssessment = evaluateDanger(
        { lat: droneLat, lng: droneLng },
        { lat: USER_POSITION.lat, lng: USER_POSITION.lng },
        15, // Vitesse estimée d'un drone FPV (m/s)
        droneHeading // Cap réel ou supposé du drone
      )

      const { level, distanceToUser, message, color, eta, threatAnalysis, angleDifference } = dangerAssessment

      // Alerte pour niveaux critiques (une seule fois par détection)
      if ((level === "critical" || level === "danger") && onDangerAlert && !alertedDetectionsRef.current.has(detection.id)) {
        alertedDetectionsRef.current.add(detection.id)
        onDangerAlert(message, level)
      }

      // Ajouter un cercle pour le drone
      const droneCircle = L.circle([droneLat, droneLng], {
        color: color,
        fillColor: color,
        fillOpacity: 0.3,
        radius: 50,
        bubblingMouseEvents: false, // Empêcher la propagation des événements
      }).addTo(map)

      // Tooltip au survol (hover) avec analyse complète
      droneCircle.bindTooltip(
        `
          <div style="font-family: system-ui, -apple-system, sans-serif;">
            <b style="font-size: 13px;">🚁 Détection Drone</b><br>
            <div style="margin-top: 4px; font-size: 12px;">
              <b>Caméra:</b> ${camera.name}<br>
              <b>Confiance:</b> ${(detection.confidence * 100).toFixed(1)}%<br>
              <b>📷 Distance caméra→drone:</b> ~${Math.round(distanceCameraToDrone)}m<br>
              <b>Direction vue:</b> ${droneBearing}° (${droneCardinal})<br>
              <b>Cap drone:</b> ${Math.round(droneHeading)}°<br>
              <hr style="margin: 6px 0; border-color: #ccc;">
              <b>👤 Distance utilisateur:</b> ${Math.round(distanceToUser)}m<br>
              <b>Déviation angulaire:</b> ${angleDifference}°<br>
              <b>Analyse:</b> ${threatAnalysis}<br>
              ${eta ? `<b>⏱️ ETA:</b> ${eta}<br>` : ""}
              <span style="color: ${color}; font-weight: bold; margin-top: 4px; display: inline-block;">${
          level === "critical" ? "🚨 CRITIQUE" : level === "danger" ? "⚠️ DANGER" : level === "warning" ? "⚠️ SURVEILLANCE" : "✅ SÛR"
        }</span>
            </div>
          </div>
        `,
        {
          permanent: false,
          direction: "top",
          offset: [0, -10],
          opacity: 0.95,
        }
      )

      // Popup au clic avec bouton de suppression
      const popupContent = `
        <div style="font-family: system-ui, -apple-system, sans-serif;">
          <b>Détection Drone</b><br>
          Caméra: ${camera.name}<br>
          Confiance: ${(detection.confidence * 100).toFixed(1)}%<br>
          📷 Distance caméra→drone: ~${Math.round(distanceCameraToDrone)}m<br>
          Direction vue: ${droneBearing}° (${droneCardinal})<br>
          Cap drone: ${Math.round(droneHeading)}°<br>
          <hr>
          👤 Distance utilisateur: ${Math.round(distanceToUser)}m<br>
          Déviation angulaire: ${angleDifference}°<br>
          Analyse: ${threatAnalysis}<br>
          ${eta ? `⏱️ ETA: ${eta}<br>` : ""}
          <span style="color: ${color}; font-weight: bold;">${
        level === "critical" ? "🚨 CRITIQUE" : level === "danger" ? "⚠️ DANGER" : level === "warning" ? "⚠️ SURVEILLANCE" : "✅ SÛR"
      }</span>
          <hr style="margin: 8px 0;">
          <button 
            id="delete-detection-btn"
            style="
              width: 100%;
              padding: 6px 12px;
              background-color: #ef4444;
              color: white;
              border: none;
              border-radius: 4px;
              cursor: pointer;
              font-weight: bold;
              font-size: 12px;
            "
            onmouseover="this.style.backgroundColor='#dc2626'"
            onmouseout="this.style.backgroundColor='#ef4444'"
          >
            🗑️ Supprimer cette détection
          </button>
        </div>
      `
      
      const popup = L.popup({
        closeButton: true,   // Afficher le bouton X pour fermer
        closeOnClick: false, // Ne pas fermer quand on clique ailleurs sur la carte
        autoClose: false,    // Ne fermer que quand on ouvre une autre popup
        closeOnEscapeKey: true, // Fermer avec Echap
      }).setContent(popupContent)
      
      droneCircle.bindPopup(popup, {
        maxWidth: 300,
      })

      // Ligne de la caméra au drone
      const cameraLine = L.polyline(
        [
          [camera.lat, camera.lng],
          [droneLat, droneLng],
        ],
        {
          color: color,
          weight: 2,
          opacity: 0.5,
          dashArray: "5, 10",
        }
      ).addTo(map)

      // Flèche indiquant la direction du drone (basée sur le cap réel du drone)
      const arrowLength = 0.0015 // degrés (~167m)
      
      // Calculer la fin de la flèche en utilisant le CAP du drone (droneHeading)
      const arrowAngleRad = (droneHeading * Math.PI) / 180
      const arrowEndLat = droneLat + Math.cos(arrowAngleRad) * arrowLength
      const arrowEndLng = droneLng + Math.sin(arrowAngleRad) * arrowLength

      // Dessiner la flèche de direction
      const arrow = L.polyline(
        [
          [droneLat, droneLng],
          [arrowEndLat, arrowEndLng],
        ],
        {
          color: color,
          weight: 4,
          opacity: 0.9,
        }
      ).addTo(map)

      // Ajouter une pointe de flèche (triangle)
      const arrowHead = L.circle([arrowEndLat, arrowEndLng], {
        color: color,
        fillColor: color,
        fillOpacity: 1,
        radius: 25,
        weight: 0,
      }).addTo(map)

      // Tooltip sur la flèche - afficher le cap du drone
      const droneCardinalHeading = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"][Math.round(droneHeading / 22.5) % 16]
      arrow.bindTooltip(`Cap drone: ${droneHeading}° (${droneCardinalHeading})`, {
        permanent: false,
        direction: "top",
        offset: [0, -5],
      })
      
      // Stocker les layers pour ce drone
      droneLayersRef.current.set(detection.id, {
        circle: droneCircle,
        line: cameraLine,
        arrow: arrow,
        arrowHead: arrowHead
      })
      
      // Mettre à jour l'événement de suppression pour supprimer tous les éléments associés
      droneCircle.on('popupopen', () => {
        // Attendre que le DOM soit prêt
        setTimeout(() => {
          const deleteButton = document.getElementById('delete-detection-btn')
          if (deleteButton) {
            deleteButton.addEventListener('click', (e) => {
              e.preventDefault()
              e.stopPropagation()
              
              // Ajouter à la blacklist
              addToBlacklist(detection.id)
              blacklistRef.current.add(detection.id)
              console.log('🚫 Ajouté à la blacklist:', detection.id)
              
              // Supprimer de la carte
              map.removeLayer(droneCircle)
              map.removeLayer(cameraLine)
              map.removeLayer(arrow)
              map.removeLayer(arrowHead)
              droneLayersRef.current.delete(detection.id)
              map.closePopup()
            })
          }
        }, 100)
      })

    })
  }, [detections, onDangerAlert])

  return (
    <div className="w-full h-full min-h-[600px] rounded-lg overflow-hidden border-2 border-zinc-700" ref={mapContainerRef} />
  )
}
