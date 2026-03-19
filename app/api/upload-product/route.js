"use server";
// app/api/upload-product/route.js
// ensures server runtime

 // important: prevent client-side bundling

import connectDB from "@/db/connectDb";
import Product from "@/models/Product";
import { index } from "@/lib/pinecone";

// -------------------- Helpers --------------------
function averageEmbedding(tensor) {
  if (!tensor || !tensor.data || !tensor.dims) return [];
  const dims = tensor.dims;
  const data = tensor.data;

  if (dims.length === 2) return Array.from(data);

  if (dims.length === 3) {
    const tokens = dims[1];
    const dim = dims[2];
    const avg = new Array(dim).fill(0);
    for (let t = 0; t < tokens; t++) {
      for (let d = 0; d < dim; d++) {
        avg[d] += data[t * dim + d];
      }
    }
    return avg.map((v) => v / tokens);
  }
  return [];
}

// -------------------- Lazy load Transformers (WASM only) --------------------
let textEmbedder, imageClassifier, llmPipeline;
async function loadPipelines() {
  if (!textEmbedder || !imageClassifier || !llmPipeline) {
    const transformers = await import("@xenova/transformers"); // dynamic import
    textEmbedder = await transformers.pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L12-v2",
      { backend: "wasm" }
    );
    imageClassifier = await transformers.pipeline(
      "zero-shot-image-classification",
      "Xenova/clip-vit-base-patch32",
      { backend: "wasm" }
    );
    llmPipeline = await transformers.pipeline(
      "text-generation",
      "Xenova/distilgpt2",
      { backend: "wasm" }
    );
  }
  return { textEmbedder, imageClassifier, llmPipeline };
}

// -------------------- API Route --------------------
export async function POST(req) {
  try {
    const { textEmbedder, imageClassifier, llmPipeline } = await loadPipelines();

    const data = await req.json();
    const { name, description, materials, ecoImpact, artistStory, culturalMeaning, price, imageBase64 } = data;

    if (!name || !description || !materials || !imageBase64) {
      return new Response(JSON.stringify({ error: "All required fields missing" }), { status: 400 });
    }

    // -------------------- Text Embedding --------------------
    const textEmbeddingRaw = await textEmbedder(description + " " + materials);
    const textEmbedding = averageEmbedding(textEmbeddingRaw);

    if (textEmbedding.length !== 384) {
      return new Response(JSON.stringify({ error: `Embedding mismatch: ${textEmbedding.length} != 384` }), { status: 500 });
    }

    // -------------------- Image Classification --------------------
    const imageBuffer = Buffer.from(imageBase64.replace(/^data:image\/\w+;base64,/, ""), "base64");
    const labels = [
      "plastic object",
      "ceramic clay pottery",
      "wooden handmade object",
      "metal object",
      "glass object",
      "eco friendly handmade product",
    ];
    const result = await imageClassifier(imageBuffer, labels);
    const plasticScore = result.find((r) => r.label === "plastic object")?.score || 0;
    const isPlastic = plasticScore > 0.4;

    let allowSubmit = !isPlastic;
    let plasticWarning = isPlastic ? `⚠️ Image suggests plastic (confidence: ${plasticScore.toFixed(2)})` : "";

    const materialsText = materials.toLowerCase();
    if (isPlastic && materialsText.includes("clay")) {
      plasticWarning += "\n⚠️ Mismatch: Image looks plastic but materials say clay.";
      allowSubmit = false;
    }

    // -------------------- Pinecone Upsert --------------------
    await index.upsert([{
      id: Date.now().toString(),
      values: textEmbedding,
      metadata: { name, description, materials, ecoImpact, artistStory, culturalMeaning, price },
    }]);

    // -------------------- LLM Explanation --------------------
    const prompt = `Explain in 2-3 short lines why this product is eco-friendly.

Product: ${description}
Materials: ${materials}`;
    const ragResult = await llmPipeline(prompt, { max_new_tokens: 60, temperature: 0.5 });
    let generatedText = Array.isArray(ragResult) ? ragResult[0].generated_text : ragResult.generated_text;
    let explanation = generatedText.replace(prompt, "").trim();

    if (!explanation || explanation.length < 20) {
      explanation = isPlastic
        ? "This product is not eco-friendly because it likely contains plastic."
        : "This product is eco-friendly as it uses natural and sustainable materials.";
    }

    // -------------------- Save to MongoDB --------------------
    if (allowSubmit) {
      await connectDB();
      await Product.create({
        name, description, materials, ecoImpact, artistStory, culturalMeaning,
        price: Number(price),
        image: imageBase64,
        ecoDecision: isPlastic ? "NOT_ECO" : "ECO"
      });
    }

    return new Response(JSON.stringify({
      success: true,
      explanation,
      allowSubmit,
      plasticInfo: { isPlastic, score: plasticScore },
      plasticWarning
    }), { status: 200 });

  } catch (error) {
    console.error("Upload Error:", error);
    return new Response(JSON.stringify({ error: error.message || "Failed to process product" }), { status: 500 });
  }
}