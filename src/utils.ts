import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import dotenv from 'dotenv';
import { Configuration, OpenAIApi } from "openai";
import { NDArray } from 'vectorious';
import * as tf from '@tensorflow/tfjs-node'
import * as sk from 'scikitjs'
import { KMeans } from 'scikitjs';
import crypto from 'crypto';
import {chunkParagraphs} from "./chunkParagraphs";


sk.setBackend(tf)

dotenv.config();

const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
    organization: process.env.OPENAI_ORGANIZATION,
});
const openai = new OpenAIApi(configuration);

function save_yaml(filepath: string, data: any): void {
    fs.promises.writeFile(filepath, yaml.dump(data), 'utf-8');
}

async function load_yaml(filepath: string): Promise<any> {
    const content = await fs.promises.readFile(filepath, 'utf-8');
    return yaml.load(content);
}

interface Message {
    content: string;
    speaker: string;
    timestamp: number;
}

export const create_message = (message: string, speaker: string, timestamp: number): Message => {
    return { content: message, speaker, timestamp };
};

export const save_message = async (root_folder: string, message: Message): Promise<void> => {
    const { timestamp, speaker } = message;
    const filename = `chat_${timestamp}_${speaker}.yaml`;
    const filepath = path.join(root_folder, 'L1_raw_logs', filename);
    fs.mkdirSync(path.dirname(filepath), { recursive: true });
    await save_yaml(filepath, message);
};

function cosine_similarity(vec1: number[], vec2: number[]): number {
    const a = new NDArray(vec1);
    const b = new NDArray(vec2);

    const dotProduct = a.dot(b);
    const magnitudeA = a.norm();
    const magnitudeB = b.norm();

    const similarity = dotProduct / (magnitudeA * magnitudeB);
    return similarity;
}

export const search_tree = async (root_folder: string, query: string): Promise<{taxonomy: string[], content: { file: string, similarity: number, content: string}[] }> => {
    const query_embeddings = await embedding_model([query]);
    const query_embedding = query_embeddings[0];
    let level = 6;
    const taxonomy: string[] = [];

    while (level > 2) {
        const level_dir = path.join(root_folder, `L${level}_summaries`);
        if (fs.existsSync(level_dir) && fs.readdirSync(level_dir).length) {
            break;
        }
        level -= 1;
    }

    let childFiles: string[] = [];

    while (level > 2) {
        const level_dir = path.join(root_folder, `L${level}_summaries`);
        const level_files = fs
            .readdirSync(level_dir)
            .filter((f) => f.endsWith('.yaml'))
            .map((f) => path.join(level_dir, f));

        let max_similarity = -1;
        let closest_file: string | null = null;

        for (const file of level_files) {
            const data = await load_yaml(file);
            const similarity = cosine_similarity(query_embedding, data.vector);
            console.log(`Similarity of ${query} to ${file}: ${similarity}`);

            if (similarity > max_similarity) {
                max_similarity = similarity;
                closest_file = file;
            }
        }

        console.log('closest_file', closest_file);
        const closest_data = await load_yaml(closest_file as string);
        taxonomy.push(closest_data.content);
        childFiles = closest_data.files as string[];

        if (level === 2) {
            break;
        }

        level -= 1;
        const child_files = closest_data.files;
        const new_level_dir = path.join(root_folder, `L${level}_summaries`);
        level_files.length = 0; // clear the array
        for (const f of child_files) {
            level_files.push(path.join(new_level_dir, f));
        }
    }

    console.log('taxonomy', taxonomy);

    let similarities: { file: string, similarity: number, content: string}[] = [];
    for (const file of childFiles) {
        const data = await load_yaml(file);
        const content = data.content as string;
        const similarity = cosine_similarity(query_embedding, data.vector);
        console.log(`Similarity of ${query} to ${file}: ${similarity}`);

        similarities.push({ file, similarity, content});
    }
    // Sort similarities
    similarities = similarities.sort((a, b) => b.similarity - a.similarity);

    // Get top 5 most similar files
    const top5 = similarities.slice(0, 5);


    return {
        taxonomy: taxonomy,
        content: top5
    };
};


async function embedding_model(texts: string[]): Promise<number[][]> {
    const cacheFolderPath = 'embedding_cache';

    const hash = crypto.createHash('sha256').update(JSON.stringify(texts)).digest('hex');
    const cacheFilePath = path.join(cacheFolderPath, `${hash}.json`);

    if (fs.existsSync(cacheFilePath)) {
        const cachedData = JSON.parse(fs.readFileSync(cacheFilePath, 'utf-8'));
        return cachedData.embeddings;
    }

    try {
        const response = await openai.createEmbedding({
            model: 'text-embedding-ada-002', // or the desired model
            input: texts,
        })

        const embeddings = response.data.data.map((embedding) => {
            return embedding.embedding;
        });

        if (!fs.existsSync(cacheFolderPath)) {
            fs.mkdirSync(cacheFolderPath);
        }

        fs.writeFileSync(cacheFilePath, JSON.stringify({ texts, embeddings }));

        return embeddings;
    } catch (error) {
        console.error('Error generating embeddings:', error);
        throw error;
    }
}

export const rebuild_tree = async (root_folder: string, max_cluster_size: number = 10) => {
    // Delete all folders except L1_raw_logs, L2_message_pairs, and .git
    for (const folder_name of fs.readdirSync(root_folder)) {
        if (!['L1_raw_logs', 'L2_message_pairs', '.git'].includes(folder_name)) {
            const folder_path = path.join(root_folder, folder_name);
            if (fs.statSync(folder_path).isDirectory()) {
                fs.rmSync(folder_path, { recursive: true });
            }
        }
    }

    // Create L2 directory if it does not exist
    const l2_message_pairs_dir = path.join(root_folder, 'L2_message_pairs');
    if (!fs.existsSync(l2_message_pairs_dir)) {
        fs.mkdirSync(l2_message_pairs_dir, { recursive: true });
    }

    // Process any missing messages in L1 to generate message pairs for L2
    await process_missing_messages(root_folder);

    // Cluster L2 message pairs using cosine similarity, up to 10 per cluster
    const clusters = await cluster_elements(root_folder, 'L2_message_pairs', max_cluster_size);

    // Create summaries and save them in the next rank (L3_summaries)
    await create_summaries(root_folder, clusters, `L3_summaries`, 'L2_message_pairs', 2);

    // If top rank (e.g. L3_summaries) has > max_cluster_size files, repeat process, creating new taxonomical ranks
    let current_rank = 3;
    while (true) {
        // Calculate clusters at new rank
        const clusters = await cluster_elements(root_folder, `L${current_rank}_summaries`, max_cluster_size);

        // Summarize those clusters
        await create_summaries(root_folder, clusters, `L${current_rank + 1}_summaries`, `L${current_rank}_summaries`, current_rank);
        current_rank += 1;

        // If clusters are less than max cluster size, we are done :)
        if (clusters.length <= max_cluster_size) {
            break;
        }
    }
};

const process_missing_messages = async (root_folder: string) => {
    const raw_logs_dir = path.join(root_folder, 'L1_raw_logs');
    const message_pairs_dir = path.join(root_folder, 'L2_message_pairs');

    // Get list of processed message filenames
    const processed_messages = new Set(fs.readdirSync(message_pairs_dir));

    // Sort raw log files by timestamp
    const raw_log_files = fs.readdirSync(raw_logs_dir);

    for (let i = 0; i < raw_log_files.length - 1; i++) {
        const file1_path = path.join(raw_logs_dir, raw_log_files[i]);
        const file2_path = path.join(raw_logs_dir, raw_log_files[i + 1]);

        // Check if message pair is already processed
        const message_pair_filename = `pair_${raw_log_files[i + 1]}`;
        if (processed_messages.has(message_pair_filename)) {
            continue;
        }

        // Load raw log data
        const file1_data = await load_yaml(file1_path);
        const file2_data = await load_yaml(file2_path);

        const context = file1_data.content;
        const response = file2_data.content;
        const speaker = file2_data.speaker;
        const timestamp = file2_data.timestamp;
        const combined_text = context + " --- " + response;
        const embedding = await embedding_model([combined_text]);

        const message_pair_data = {
            content: combined_text,
            speaker: speaker,
            timestamp: timestamp,
            vector: embedding[0]
        };

        // Save message pair in L2_message_pairs folder
        const message_pair_path = path.join(message_pairs_dir, message_pair_filename);
        await save_yaml(message_pair_path, message_pair_data);
    }
}


async function create_summaries(root_folder: string, clusters: string[][], target_folder: string, source_folder: string, currentRank: number): Promise<void> {
    const source_folder_path = path.join(root_folder, source_folder);
    const target_folder_path = path.join(root_folder, target_folder);
    fs.mkdirSync(target_folder_path, { recursive: true });

    for (let i = 0; i < clusters.length; i++) {
        const cluster = clusters[i];

        // Combine content of cluster elements
        let combined_content = "";
        const files = [];
        for (const file of cluster) {
            const filepath = path.join(source_folder_path, file);
            const data = await load_yaml(filepath);
            combined_content += data["content"] + " \n---\n ";
            files.push(filepath);
        }

        // Generate summary with LLM
        const summary = await quick_summarize(combined_content, currentRank);

        // Create embedding for summary
        const summary_embedding = await embedding_model([summary]); // Assuming embedding_model is a Promise-based function

        // Save summary in target folder
        const summary_data = {
            content: summary,
            vector: summary_embedding[0],
            files: files,
            timestamp: Date.now()
        };

        const timestamp = Date.now();
        const summary_filename = `summary_${i}_${timestamp}.yaml`;
        const summary_filepath = path.join(target_folder_path, summary_filename);
        await save_yaml(summary_filepath, summary_data);
    }
}

async function cluster_elements(root_folder: string, target_folder: string, max_cluster_size: number = 2): Promise<string[][]> {
    const folder_path = path.join(root_folder, target_folder);
    const yaml_files = fs.readdirSync(folder_path).filter((file) => file.endsWith(".yaml"));

    // Load vectors
    const vectors: number[][] = [];
    for (const file of yaml_files) {
        const filepath = path.join(folder_path, file);
        const data = await load_yaml(filepath);
        vectors.push(data["vector"]);
    }

    if (vectors.length === 0) {
        console.log("No vectors to cluster. Files: ", yaml_files, " Folder: ", folder_path);
        return [];
    }

    // Calculate number of clusters
    const num_clusters = Math.ceil(yaml_files.length / max_cluster_size);

    // Instantiate KMeans with the appropriate options
    const kmeans = new KMeans({ nClusters: num_clusters, randomState: 42 });

    // Fit the model to the data
    const result = kmeans.fitPredict(vectors);
    const labels = await result.array();

    // Group files by cluster
    const clusters: string[][] = [];
    for (let i = 0; i < labels.length; i++) {
        const label = labels[i];
        if (!clusters[label]) {
            clusters[label] = [];
        }
        clusters[label].push(yaml_files[i]);
    }

    return clusters;
}

export async function maintain_tree(root_folder: string): Promise<void> {
    const l2_message_pairs_dir = path.join(root_folder, "L2_message_pairs");

    // Create L2 directory if it does not exist
    if (!fs.existsSync(l2_message_pairs_dir)) {
        fs.mkdirSync(l2_message_pairs_dir, { recursive: true });
    }

    // Get list of files in L2 before processing missing messages
    const l2_files_before = new Set(fs.readdirSync(l2_message_pairs_dir));

    // Process missing messages to generate new message pairs in L2
    await process_missing_messages(root_folder);

    // Get list of files in L2 after processing missing messages
    const l2_files_after = new Set(fs.readdirSync(l2_message_pairs_dir));

    // Calculate the difference between the two lists to obtain the new message pairs
    const new_message_pairs = new Set([...l2_files_after].filter((file) => !l2_files_before.has(file)));

    // Iterate through new files in L2 and check cosine similarity to files in L3
    await integrate_new_elements(root_folder, "L3_summaries", new_message_pairs, 0.75);
}


async function integrate_new_elements(root_folder: string, target_folder: string, new_elements: Set<string>, threshold: number): Promise<void> {
    const target_dir = path.join(root_folder, target_folder);

    // Create target directory if it does not exist
    if (!fs.existsSync(target_dir)) {
        fs.mkdirSync(target_dir, { recursive: true });
    }

    for (const new_element of new_elements) {
        const new_element_path = path.join(root_folder, "L2_message_pairs", new_element);
        const new_element_data = await load_yaml(new_element_path);
        const new_element_vector = new_element_data["vector"];

        let max_similarity = -1;
        let closest_file = null;

        for (const file of fs.readdirSync(target_dir)) {
            const file_path = path.join(target_dir, file);
            const file_data = await load_yaml(file_path);
            const file_vector = file_data["vector"];

            const similarity = cosine_similarity(new_element_vector, file_vector);
            if (similarity > max_similarity) {
                max_similarity = similarity;
                closest_file = file;
            }
        }

        if (max_similarity > threshold) {
            // Update the corresponding summary and record the name of the modified file
            const closest_file_path = path.join(target_dir, closest_file);
            const closest_file_data = await load_yaml(closest_file_path);
            closest_file_data["files"].push(new_element);

            const combined_content = closest_file_data["content"] + " --- " + new_element_data["content"];
            const updated_summary = await quick_summarize(combined_content, 2);
            const updated_summary_embedding = await embedding_model([updated_summary]);

            closest_file_data["content"] = updated_summary;
            closest_file_data["vector"] = updated_summary_embedding[0];
            closest_file_data["timestamp"] = Date.now() / 1000;

            await save_yaml(closest_file_path, closest_file_data);
        } else {
            // Create a new summary for the new_element
            const combined_content = new_element_data["content"];
            const new_summary = await quick_summarize(combined_content, 2);
            const new_summary_embedding = await embedding_model([new_summary]);

            const new_summary_data = {
                "content": new_summary,
                "vector": new_summary_embedding[0],
                "files": [new_element],
                "timestamp": Date.now() / 1000,
            };

            const new_summary_filename = `summary_${fs.readdirSync(target_dir).length}.yaml`;
            const new_summary_filepath = path.join(target_dir, new_summary_filename);
            await save_yaml(new_summary_filepath, new_summary_data);
        }
    }
}

function getSummarizePrompt(text: string, rank:number): string {
    if (rank == 2) {
        return `Extract the main points of this text in a bulleted list:\n${text}`;
    } else {
        return `Extract a list of subjects the following points relate to:\n${text}`;
    }
}

async function quick_summarize(text: string, currentRank:number): Promise<string> {
    const chunkTokenSize = 3500;
    const { chunks } = await chunkParagraphs({ text, chunkTokenSize });

    const summaries = [];
    for (const chunk of chunks) {
        const prompt = getSummarizePrompt(chunk, currentRank);
        const response = await gpt3_completion(prompt);
        summaries.push(response);
    }

    const final_summary = summaries.join(' ');
    return final_summary;
}

async function gpt3_completion(
    prompt: string,
    temp = 0.0,
    top_p = 1.0,
    tokens = 500,
    freq_pen = 0.0,
    pres_pen = 0.0,
    stop: string[] = ['asdfasdfasdf']
): Promise<string> {
    const max_retry = 5;
    let retry = 0;
    prompt = prompt.replace(/[^\x00-\x7F]/g, '');

    const hash = crypto.createHash('sha256').update(prompt).digest('hex');
    const cacheFolderPath = 'gpt_cache';
    const cacheFilePath = path.join(cacheFolderPath, `${hash}.json`);

    if (fs.existsSync(cacheFilePath)) {
        const cachedData = JSON.parse(fs.readFileSync(cacheFilePath, 'utf-8'));
        return cachedData.response;
    }

    while (true) {
        try {
            let response = await openai.createChatCompletion({
                model: 'gpt-3.5-turbo',
                messages: [
                    {
                        role: "user",
                        content: prompt,
                    }
                ],
                temperature: temp,
                max_tokens: tokens,
                top_p: top_p,
                frequency_penalty: freq_pen,
                presence_penalty: pres_pen,
                stop: stop,
            });
            const text = response.data.choices[0]?.message?.content ?? '';

            if (!fs.existsSync(cacheFolderPath)) {
                fs.mkdirSync(cacheFolderPath);
            }

            fs.writeFileSync(cacheFilePath, JSON.stringify({ prompt, response: text }));

            return text;
        } catch (error) {
            retry += 1;
            if (retry >= max_retry) {
                return `GPT error: ${error}`;
            }
            console.log('Error communicating with OpenAI:', error);
            await new Promise((resolve) => setTimeout(resolve, 1000));
        }
    }
}
