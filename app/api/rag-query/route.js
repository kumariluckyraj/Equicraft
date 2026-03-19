// File: app/api/rag-query/route.js
"use server";
// ensures server runtime

 // tells Next.js not to bundle this in client code

import fs from "fs";
import path from "path";

// Cosine similarity helper
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}

// Lazily load WASM-only transformers
let llmPipeline, textEmbedderPipeline;
async function loadPipelines() {
  if (!llmPipeline || !textEmbedderPipeline) {
    const { pipeline } = await import("@xenova/transformers"); // 🔹 Dynamic import at runtime
    textEmbedderPipeline = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { backend: "wasm" });
    llmPipeline = await pipeline("text-generation", "Xenova/gpt2", { backend: "wasm" });
  }
  return { llmPipeline, textEmbedderPipeline };
}

export async function POST(req) {
  try {
    const { question } = await req.json();
    if (!question) return new Response(JSON.stringify({ error: "Question required" }), { status: 400 });

    // Load vector store
    const storePath = path.join(process.cwd(), "vector-store/products.json");
    if (!fs.existsSync(storePath)) return new Response(JSON.stringify({ error: "No products found" }), { status: 404 });
    const products = JSON.parse(fs.readFileSync(storePath, "utf-8"));

    // Load pipelines dynamically
    const { llmPipeline, textEmbedderPipeline } = await loadPipelines();

    // 1️⃣ Embed question
    const questionEmbeddingRaw = await textEmbedderPipeline(question);
    const questionEmbedding = questionEmbeddingRaw.flat();

    // 2️⃣ Rank top 3 products
    const ranked = products
      .map((p) => ({ ...p, score: cosineSimilarity(questionEmbedding, p.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    // 3️⃣ Build context
    const contextText = ranked
      .map(
        (p, i) =>
          `Product ${i + 1}:\nName: ${p.name}\nDescription: ${p.description}\nMaterials: ${p.materials}\nEcoImpact: ${p.ecoImpact}\nArtistStory: ${p.artistStory}\nCulturalMeaning: ${p.culturalMeaning}`
      )
      .join("\n\n");

    const prompt = `You are a sustainable craft expert. Answer the following question using only the products listed below. Be concise and informative.\n\n${contextText}\n\nQuestion: ${question}\nAnswer:`;

    // 4️⃣ Generate answer
    const llmResult = await llmPipeline(prompt, { max_new_tokens: 200 });
    const answer = Array.isArray(llmResult) ? llmResult[0].generated_text : llmResult.generated_text;

    return new Response(JSON.stringify({ answer: answer.trim(), topProducts: ranked }), { status: 200 });
  } catch (err) {
    console.error("RAG Query Error:", err);
    return new Response(JSON.stringify({ error: "Failed to query products" }), { status: 500 });
  }
}