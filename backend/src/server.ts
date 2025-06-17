// backend/src/server.ts - Hauptserver mit allen Routen
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { PrismaClient } from '@prisma/client';
import dotenv from 'dotenv';
import { createServer } from 'http';
import { Server } from 'socket.io';

// Import Controllers
import { ProjectController } from './controllers/projectController';
import { AnalysisController } from './controllers/analysisController';
import { PriceController } from './controllers/priceController';
import { ExportController } from './controllers/exportController';

// Import Services
import { AIService } from './services/aiService';
import { ScraperService } from './services/scraperService';
import { PriceCalculationService } from './services/priceService';
import { GAEBService } from './services/gaebService';

// Import Middleware
import { authMiddleware } from './middleware/auth';
import { errorHandler } from './middleware/errorHandler';
import { rateLimiter } from './middleware/rateLimiter';

dotenv.config();

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: process.env.FRONTEND_URL,
    credentials: true
  }
});

const prisma = new PrismaClient();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL,
  credentials: true
}));
app.use(express.json({ limit: '50mb' }));
app.use(morgan('combined'));
app.use('/api', rateLimiter);

// Initialize Services
const aiService = new AIService();
const scraperService = new ScraperService();
const priceService = new PriceCalculationService(prisma);
const gaebService = new GAEBService();

// Initialize Controllers
const projectController = new ProjectController(prisma, io);
const analysisController = new AnalysisController(prisma, aiService, scraperService, io);
const priceController = new PriceController(prisma, priceService, aiService, io);
const exportController = new ExportController(prisma, gaebService);

// Routes
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Project Routes
app.post('/api/projects', authMiddleware, projectController.create);
app.get('/api/projects', authMiddleware, projectController.list);
app.get('/api/projects/:id', authMiddleware, projectController.get);
app.put('/api/projects/:id', authMiddleware, projectController.update);
app.delete('/api/projects/:id', authMiddleware, projectController.delete);

// Analysis Routes
app.post('/api/analyze/tender', authMiddleware, analysisController.analyzeTender);
app.post('/api/analyze/document', authMiddleware, analysisController.analyzeDocument);
app.get('/api/analyze/status/:jobId', authMiddleware, analysisController.getJobStatus);

// Price Calculation Routes
app.post('/api/prices/calculate', authMiddleware, priceController.calculatePrices);
app.post('/api/prices/optimize', authMiddleware, priceController.optimizePrices);
app.get('/api/prices/history', authMiddleware, priceController.getPriceHistory);
app.get('/api/prices/statistics', authMiddleware, priceController.getStatistics);

// Export Routes
app.post('/api/export/gaeb', authMiddleware, exportController.exportGAEB);
app.post('/api/export/pdf', authMiddleware, exportController.exportPDF);
app.post('/api/export/excel', authMiddleware, exportController.exportExcel);

// WebSocket Events
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  socket.on('join-project', (projectId) => {
    socket.join(`project-${projectId}`);
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Error Handler
app.use(errorHandler);

// Start Server
httpServer.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// ===== SERVICE IMPLEMENTATIONS =====

// backend/src/services/aiService.ts
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';

interface ExtractedData {
  projektName: string;
  ort: string;
  zeitraum: string;
  beschreibung: string;
  gewerke: string[];
  leistungen: Array<{
    beschreibung: string;
    menge?: number;
    einheit?: string;
  }>;
}

export class AIService {
  private openai: OpenAI;
  private anthropic: Anthropic;
  
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
    
    this.anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
  }
  
  async extractTenderData(text: string): Promise<ExtractedData> {
    const systemPrompt = `
    Du bist ein Experte für die Analyse von Bauausschreibungen.
    Extrahiere folgende Informationen aus dem Text:
    1. Projektname
    2. Ort der Durchführung
    3. Zeitraum/Fristen
    4. Projektbeschreibung
    5. Gewerke (z.B. Malerarbeiten, Trockenbau, etc.)
    6. Konkrete Leistungen mit Mengenangaben wenn vorhanden
    
    Gib die Antwort als strukturiertes JSON zurück.
    `;
    
    try {
      const completion = await this.openai.chat.completions.create({
        model: "gpt-4-turbo-preview",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: text }
        ],
        response_format: { type: "json_object" },
        temperature: 0.3,
      });
      
      return JSON.parse(completion.choices[0].message.content || '{}');
    } catch (error) {
      console.error('OpenAI Fehler, verwende Claude:', error);
      
      // Fallback zu Claude
      const message = await this.anthropic.messages.create({
        model: 'claude-3-opus-20240229',
        max_tokens: 4000,
        messages: [{
          role: 'user',
          content: `${systemPrompt}\n\nText:\n${text}`
        }]
      });
      
      const content = message.content[0].text;
      return JSON.parse(content);
    }
  }
  
  async generateLVPositions(projectData: ExtractedData): Promise<any[]> {
    const prompt = `
    Erstelle ein detailliertes Leistungsverzeichnis basierend auf folgenden Projektdaten:
    ${JSON.stringify(projectData, null, 2)}
    
    Verwende die DIN 276 Kostengliederung.
    Jede Position muss enthalten:
    - Positionsnummer (hierarchisch, z.B. 01.01.010)
    - Kurzbeschreibung (max. 80 Zeichen)
    - Detaillierte Leistungsbeschreibung
    - Menge (realistische Schätzung)
    - Einheit (m², m³, Stk, etc.)
    - Gewerk/Kostengruppe
    
    Format: JSON-Array
    `;
    
    const completion = await this.openai.chat.completions.create({
      model: "gpt-4-turbo-preview",
      messages: [
        { role: "system", content: "Du bist ein erfahrener Baukalkulator." },
        { role: "user", content: prompt }
      ],
      response_format: { type: "json_object" },
      temperature: 0.5,
    });
    
    const result = JSON.parse(completion.choices[0].message.content || '{}');
    return result.positions || [];
  }
  
  async estimatePrice(position: any, region: string, strategy: string): Promise<number> {
    const prompt = `
    Schätze einen realistischen Einheitspreis für folgende Bauleistung:
    
    Leistung: ${position.kurzbeschreibung}
    Details: ${position.langbeschreibung}
    Einheit: ${position.einheit}
    Region: ${region}
    Preisstrategie: ${strategy}
    
    Berücksichtige:
    - Aktuelle Marktpreise 2025
    - Regionale Unterschiede
    - Materialkosten
    - Lohnkosten
    - Übliche Gewinnmargen
    
    Antworte NUR mit der Zahl (Preis in Euro), keine Währung oder Text.
    `;
    
    const completion = await this.openai.chat.completions.create({
      model: "gpt-4-turbo-preview",
      messages: [
        { role: "system", content: "Du bist ein Experte für Baupreise in Deutschland." },
        { role: "user", content: prompt }
      ],
      temperature: 0.3,
      max_tokens: 10,
    });
    
    const priceText = completion.choices[0].message.content || '0';
    return parseFloat(priceText.replace(/[^\d.]/g, ''));
  }
}

// backend/src/services/scraperService.ts
import puppeteer from 'puppeteer';
import * as cheerio from 'cheerio';

export class ScraperService {
  private browser: puppeteer.Browser | null = null;
  
  async initialize() {
    this.browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
  }
  
  async scrapeEVergabe(url: string): Promise<string> {
    if (!this.browser) await this.initialize();
    
    const page = await this.browser!.newPage();
    try {
      await page.goto(url, { waitUntil: 'networkidle2' });
      
      // Warte auf spezifische Elemente
      await page.waitForSelector('.tender-details', { timeout: 10000 });
      
      // Extrahiere Text
      const content = await page.evaluate(() => {
        const removeScripts = (element: Element) => {
          const scripts = element.querySelectorAll('script, style');
          scripts.forEach(script => script.remove());
        };
        
        const mainContent = document.querySelector('.tender-details, .main-content, #content');
        if (mainContent) {
          removeScripts(mainContent);
          return mainContent.textContent || '';
        }
        return document.body.textContent || '';
      });
      
      return this.cleanText(content);
    } finally {
      await page.close();
    }
  }
  
  async scrapeTED(url: string): Promise<string> {
    // TED-spezifische Implementierung
    const response = await fetch(url);
    const html = await response.text();
    const $ = cheerio.load(html);
    
    // TED-spezifische Selektoren
    const title = $('.document-title').text();
    const description = $('.description').text();
    const details = $('.notice-details').text();
    
    return this.cleanText(`${title}\n${description}\n${details}`);
  }
  
  async scrapeVergabe24(url: string): Promise<string> {
    // Vergabe24-spezifische Implementierung
    if (!this.browser) await this.initialize();
    
    const page = await this.browser!.newPage();
    try {
      await page.goto(url, { waitUntil: 'networkidle2' });
      
      const content = await page.evaluate(() => {
        const elements = [
          '.vergabe-titel',
          '.vergabe-beschreibung',
          '.leistungsbeschreibung',
          '.fristen'
        ];
        
        return elements
          .map(selector => {
            const el = document.querySelector(selector);
            return el ? el.textContent : '';
          })
          .join('\n');
      });
      
      return this.cleanText(content);
    } finally {
      await page.close();
    }
  }
  
  private cleanText(text: string): string {
    return text
      .replace(/\s+/g, ' ')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }
  
  async cleanup() {
    if (this.browser) {
      await this.browser.close();
    }
  }
}

// backend/src/services/priceService.ts
import { PrismaClient } from '@prisma/client';

interface PriceFactors {
  region: string;
  strategy: 'günstig' | 'marktüblich' | 'premium';
  complexity: 'einfach' | 'mittel' | 'komplex';
  urgency: boolean;
}

export class PriceCalculationService {
  constructor(private prisma: PrismaClient) {}
  
  async calculatePrice(
    position: any,
    factors: PriceFactors
  ): Promise<{ price: number; confidence: number }> {
    // 1. Historische Preise abrufen
    const historicalPrices = await this.getHistoricalPrices(
      position.kurzbeschreibung,
      factors.region
    );
    
    // 2. Basierspreis ermitteln
    let basePrice = 0;
    let confidence = 0;
    
    if (historicalPrices.length > 0) {
      // Durchschnitt der historischen Preise
      basePrice = historicalPrices.reduce((sum, p) => sum + p.price, 0) / historicalPrices.length;
      confidence = 0.9;
    } else {
      // Fallback: Standardpreise nach Gewerk
      basePrice = this.getStandardPrice(position.gewerk, position.einheit);
      confidence = 0.7;
    }
    
    // 3. Faktoren anwenden
    basePrice = this.applyFactors(basePrice, factors);
    
    // 4. Regionale Anpassung
    basePrice = this.applyRegionalFactor(basePrice, factors.region);
    
    return {
      price: Math.round(basePrice * 100) / 100,
      confidence
    };
  }
  
  private async getHistoricalPrices(description: string, region: string) {
    return await this.prisma.priceHistory.findMany({
      where: {
        OR: [
          { description: { contains: description, mode: 'insensitive' } },
          { position_type: { contains: description, mode: 'insensitive' } }
        ],
        region: region
      },
      take: 10,
      orderBy: { recorded_at: 'desc' }
    });
  }
  
  private getStandardPrice(gewerk: string, einheit: string): number {
    const standardPrices: Record<string, Record<string, number>> = {
      'Trockenbau': {
        'm²': 45,
        'm': 25,
        'Stk': 150
      },
      'Malerarbeiten': {
        'm²': 12,
        'm': 8,
        'Stk': 50
      },
      'Bodenbeläge': {
        'm²': 55,
        'm': 35,
        'Stk': 100
      },
      'Elektro': {
        'Stk': 85,
        'm': 45,
        'PA': 500
      }
    };
    
    return standardPrices[gewerk]?.[einheit] || 50;
  }
  
  private applyFactors(price: number, factors: PriceFactors): number {
    const strategyMultipliers = {
      'günstig': 0.85,
      'marktüblich': 1.0,
      'premium': 1.15
    };
    
    const complexityMultipliers = {
      'einfach': 0.9,
      'mittel': 1.0,
      'komplex': 1.2
    };
    
    price *= strategyMultipliers[factors.strategy];
    price *= complexityMultipliers[factors.complexity];
    
    if (factors.urgency) {
      price *= 1.1; // 10% Aufschlag für Eilaufträge
    }
    
    return price;
  }
  
  private applyRegionalFactor(price: number, region: string): number {
    const regionalFactors: Record<string, number> = {
      'München': 1.25,
      'Frankfurt': 1.18,
      'Stuttgart': 1.15,
      'Hamburg': 1.12,
      'Berlin': 1.08,
      'Köln': 1.10,
      'Leipzig': 0.92,
      'Dresden': 0.90,
      'Dortmund': 0.95
    };
    
    const factor = regionalFactors[region] || 1.0;
    return price * factor;
  }
}

// backend/src/controllers/analysisController.ts
import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { AIService } from '../services/aiService';
import { ScraperService } from '../services/scraperService';
import { Server } from 'socket.io';
import Bull from 'bull';

const analysisQueue = new Bull('analysis-queue', {
  redis: process.env.REDIS_URL
});

export class AnalysisController {
  constructor(
    private prisma: PrismaClient,
    private aiService: AIService,
    private scraperService: ScraperService,
    private io: Server
  ) {
    this.setupQueueProcessing();
  }
  
  private setupQueueProcessing() {
    analysisQueue.process(async (job) => {
      const { projectId, url, text } = job.data;
      
      try {
        // Progress: Scraping
        await job.progress(10);
        this.io.to(`project-${projectId}`).emit('analysis-progress', {
          stage: 'scraping',
          progress: 10
        });
        
        let content = text;
        if (url && !text) {
          content = await this.scrapeContent(url);
        }
        
        // Progress: AI Analysis
        await job.progress(40);
        this.io.to(`project-${projectId}`).emit('analysis-progress', {
          stage: 'analyzing',
          progress: 40
        });
        
        const extractedData = await this.aiService.extractTenderData(content);
        
        // Progress: LV Generation
        await job.progress(70);
        this.io.to(`project-${projectId}`).emit('analysis-progress', {
          stage: 'generating',
          progress: 70
        });
        
        const positions = await this.aiService.generateLVPositions(extractedData);
        
        // Save to database
        await this.saveAnalysisResults(projectId, extractedData, positions);
        
        // Complete
        await job.progress(100);
        this.io.to(`project-${projectId}`).emit('analysis-complete', {
          extractedData,
          positions
        });
        
        return { success: true, projectId };
      } catch (error) {
        this.io.to(`project-${projectId}`).emit('analysis-error', {
          error: error.message
        });
        throw error;
      }
    });
  }
  
  analyzeTender = async (req: Request, res: Response) => {
    try {
      const { projectId, url, text } = req.body;
      const userId = req.user.id;
      
      // Validierung
      if (!projectId || (!url && !text)) {
        return res.status(400).json({
          error: 'ProjectId und entweder URL oder Text erforderlich'
        });
      }
      
      // Job in Queue einreihen
      const job = await analysisQueue.add({
        projectId,
        url,
        text,
        userId
      });
      
      res.json({
        jobId: job.id,
        status: 'processing',
        message: 'Analyse gestartet'
      });
    } catch (error) {
      console.error('Analyse-Fehler:', error);
      res.status(500).json({ error: 'Analysefehler' });
    }
  };
  
  getJobStatus = async (req: Request, res: Response) => {
    try {
      const { jobId } = req.params;
      const job = await analysisQueue.getJob(jobId);
      
      if (!job) {
        return res.status(404).json({ error: 'Job nicht gefunden' });
      }
      
      const state = await job.getState();
      const progress = job.progress();
      
      res.json({
        jobId,
        state,
        progress,
        result: job.returnvalue
      });
    } catch (error) {
      res.status(500).json({ error: 'Fehler beim Abrufen des Job-Status' });
    }
  };
  
  private async scrapeContent(url: string): Promise<string> {
    if (url.includes('evergabe')) {
      return await this.scraperService.scrapeEVergabe(url);
    } else if (url.includes('ted.europa')) {
      return await this.scraperService.scrapeTED(url);
    } else if (url.includes('vergabe24')) {
      return await this.scraperService.scrapeVergabe24(url);
    } else {
      throw new Error('Nicht unterstützte Ausschreibungsplattform');
    }
  }
  
  private async saveAnalysisResults(
    projectId: string,
    extractedData: any,
    positions: any[]
  ) {
    // Update project
    await this.prisma.project.update({
      where: { id: projectId },
      data: {
        name: extractedData.projektName,
        location: extractedData.ort,
        timeframe: extractedData.zeitraum,
        metadata: extractedData,
        status: 'analyzed'
      }
    });
    
    // Save positions
    const positionsData = positions.map(pos => ({
      project_id: projectId,
      position_number: pos.positionsnummer,
      short_description: pos.kurzbeschreibung,
      long_description: pos.langbeschreibung,
      quantity: pos.menge,
      unit: pos.einheit,
      trade: pos.gewerk,
      confidence_score: 0.85
    }));
    
    await this.prisma.lv_position.createMany({
      data: positionsData
    });
  }
}
