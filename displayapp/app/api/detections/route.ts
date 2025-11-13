import { NextRequest, NextResponse } from 'next/server';

export interface Detection {
  id: string;
  timestamp: number;
  filename: string;
  confidence: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
  camera_id?: string;
  camera_location?: {
    lat: number;
    lng: number;
    name: string;
  };
  image_dimensions?: {
    width: number;
    height: number;
  };
  drone_heading?: number; // Cap du drone en degrés (0-360°), optionnel
  drone_location?: {      // Position GPS du drone, optionnelle
    lat: number;
    lng: number;
  };
  image_original: string;
  image_annotated: string;
  receivedAt: string;
}

// Store en mémoire (pour démo)
// En prod, utiliser Redis ou une base de données
let detections: Detection[] = [];

// Configuration - Max détections stockées en mémoire
const MAX_DETECTIONS = parseInt(process.env.MAX_DETECTIONS_STORED || '100', 10);

/**
 * POST /api/detections
 * Reçoit les détections depuis le script Python
 */
export async function POST(req: NextRequest) {
  try {
    const data = await req.json();
    
    // Validation basique
    if (!data.filename || !data.confidence || !data.bbox || !data.image_annotated) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields' },
        { status: 400 }
      );
    }
    
    // Créer la détection avec metadata
    const detection: Detection = {
      ...data,
      receivedAt: new Date().toISOString(),
      id: `det_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };
    
    // Ajouter au début du tableau (plus récent en premier)
    detections.unshift(detection);
    
    // Garder max N détections en mémoire (configurable via .env)
    if (detections.length > MAX_DETECTIONS) {
      detections = detections.slice(0, MAX_DETECTIONS);
    }
    
    console.log(`✅ Détection reçue: ${data.filename} (${data.confidence}%)`);
    
    return NextResponse.json({ 
      success: true, 
      id: detection.id,
      message: 'Detection stored successfully'
    });
    
  } catch (error) {
    console.error('❌ Erreur lors de la réception:', error);
    return NextResponse.json(
      { success: false, error: 'Invalid payload' },
      { status: 400 }
    );
  }
}

/**
 * GET /api/detections
 * Récupère toutes les détections stockées
 */
export async function GET() {
  return NextResponse.json({
    detections,
    count: detections.length,
    timestamp: new Date().toISOString()
  });
}

/**
 * DELETE /api/detections
 * Efface toutes les détections
 */
export async function DELETE() {
  const count = detections.length;
  detections = [];
  
  console.log(`🗑️  ${count} détections supprimées`);
  
  return NextResponse.json({
    success: true,
    message: `${count} detections cleared`
  });
}
