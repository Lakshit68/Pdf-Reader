import cors from 'cors';
import "dotenv/config";
import express from 'express';
import multer from 'multer';
import fs from 'fs';

import { GoogleGenAI } from "@google/genai";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { QdrantVectorStore } from '@langchain/qdrant';
import { Queue } from 'bullmq';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import pdfParse from 'pdf-parse';

async function processPDFandStoreInQdrant(pdfPath) {
  const dataBuffer = fs.readFileSync(pdfPath);
  const pdfData = await pdfParse(dataBuffer);

  const rawText = pdfData.text;

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });

  const splitDocs = await splitter.createDocuments([rawText]);

  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: 'Xenova/all-MiniLM-L6-v2',
  });

  await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
    url: process.env.qdrant_url,
    apiKey:process.env.qdrant_apikey,
    collectionName: 'langchainjs-testing', // make sure this matches the collection you retrieve from later
  });

  console.log("Stored PDF content in Qdrant.");
}

const queue = new Queue('file-upload-queue', {
  connection: {
    host: process.env.valkey_host,
    port: process.env.valkey_port,
  },
});

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, `${uniqueSuffix}-${file.originalname}`);
  },
});

const upload = multer({ storage });

const app = express();
app.use(cors());
app.use(express.json()); // To support req.body

app.get('/', (req, res) => {
  return res.json({ status: 'All Good!' });
});

// File upload route
app.post('/upload/pdf', upload.single('pdf'), async (req, res) => {
  await queue.add(
    'file-ready',
    {
      filename: req.file.originalname,
      destination: req.file.destination,
      path: req.file.path,
    }
  );
  console.log("filepath:", req.file.path)
  processPDFandStoreInQdrant(req.file.path)
  return res.json({ message: 'uploaded' });
});


// Initialize Gemini
const ai = new GoogleGenAI(process.env.GEMINI_API_KEY); // Replace with your Gemini API key

app.post('/chat', async (req, res) => {
  const userQuery = req.body.userQuery;
  console.log(userQuery);

  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: 'Xenova/all-MiniLM-L6-v2',
  });
  console.log("got embeddings");

  const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: process.env.qdrant_url,
    collectionName: 'langchainjs-testing',
  });
  console.log("got vector store");

  const retriever = vectorStore.asRetriever({ k: 2 });
  const docs = await retriever.invoke(userQuery);
  console.log("retrieved from vector store:", docs);

  const SYSTEM_PROMPT = `
You are a helpful AI Assistant who answers the user query based on the available context from the PDF file.

Context:
${JSON.stringify(docs)}
`;

  const fullPrompt = `${SYSTEM_PROMPT}\nUser: ${userQuery}\nAssistant:`;
  console.log("created full prompt");

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.0-flash-001',
      contents: fullPrompt,
    });

    const text = response.text;

    console.log("got chat result:", text);

    return res.json({
      message: text,
      docs: docs,
    });

  } catch (err) {
    console.error("Error with Gemini:", err);
    return res.status(500).json({ error: "Failed to get response from Gemini model" });
  }
});

// Chat route
// app.post('/chat', async (req, res) => {
//   const userQuery = req.body.userQuery;
//   console.log(userQuery)

//   // Use HuggingFace Transformers for embeddings
//   const embeddings = new HuggingFaceTransformersEmbeddings({
//     modelName: "Xenova/all-MiniLM-L6-v2", // Local WebAssembly model
//   });
//   console.log("got embeddings")
  
//   // Load vector store from Qdrant
//   const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
//     url: 'http://localhost:6333',
//     collectionName: 'langchainjs-testing',
//   });
//   console.log("got vector store")
  
//   const retriever = vectorStore.asRetriever({ k: 2 });
//   console.log("got retriever")
//   const docs = await retriever.invoke(userQuery, {recursionLimit: 2});
//   console.log("retrieved from vector store")
  
//   const SYSTEM_PROMPT = `
// You are a helpful AI Assistant who answers the user query based on the available context from the PDF file.

// Context:
// ${JSON.stringify(docs)}
// `;
//   console.log("created system prompt")

//   const fullPrompt = `${SYSTEM_PROMPT}\nUser: ${userQuery}\nAssistant:`;
//   console.log("created full prompt")

//   genai.configure(api_key="YOUR_GEMINI_API_KEY")
//   model = genai.GenerativeModel("gemini-pro")
//   // const llm = new HuggingFaceInference({
//   //   model: "Xenova/llama2.c-stories15M", // or any hosted model on Hugging Face Hub
//   //   apiKey: process.env.HUGGINGFACE_API_KEY,
//   //   maxTokens: 200,
//   // });
//   // console.log("got llm")
//   // const chatResult = await llm.invoke(fullPrompt);
//   // console.log("got chat result")
  
//   // const generator = await pipeline("text-generation", "Xenova/llama2.c-stories15M")
//   // // Generate the response
//   // const output = await generator(fullPrompt, {
//   //   max_new_tokens: 200, // You can adjust this
//   //   // Add other generation parameters as needed
//   // });
//   // console.log("got chat result");

//   // // The output format might vary slightly depending on the model,
//   // // but for text generation it's typically an array with a generated_text field
//   // const chatResult = output[0]?.generated_text || "Could not generate a response.";

//   // const model = new HuggingFaceTransformers({
//   //   modelName: "Xenova/llama2-7b-chat",
//   //   maxLength: 200,
//   // });
//   // console.log("got model")

//   // const chatResult = await model.call(fullPrompt);
//   // console.log("got chat result")

//   return res.json({
//     message: chatResult,
//     docs: docs,
//   });
// });
const port=process.env.PORT || 10000
app.listen(port, () => console.log(`Server started on PORT:8080`));
