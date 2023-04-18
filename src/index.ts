import fastify from 'fastify';
import { FastifyInstance } from 'fastify';
import cors from '@fastify/cors'
import * as utils from './utils';
import * as path from 'path';
import * as os from 'os';

const app = fastify({ logger: true });

app.register(cors, { origin: '*' });

const rootFolder = os.homedir();
const dataFolder = path.join(rootFolder, 'data');
const maxClusterSize = 5;

interface AddMessageInput {
    message: string;
    speaker: string;
    timestamp: number;
}

app.post<{ Body: AddMessageInput }>('/add_message', async (request, reply) => {
    const { message, speaker, timestamp } = request.body;

    const newMessage = utils.create_message(message, speaker, timestamp);
    console.log('\n\nADD MESSAGE -', newMessage);
    await utils.save_message(dataFolder, newMessage);

    return { detail: 'Message added' };
});

app.get<{ Querystring: { query: string } }>('/search', async (request, reply) => {
    const { query } = request.query;

    console.log('\n\nSEARCH -', query);
    const taxonomy = await utils.search_tree(dataFolder, query);

    return { results: taxonomy };
});

app.post('/rebuild_tree', async (request, reply) => {
    console.log('\n\nREBUILD TREE');
    await utils.rebuild_tree(dataFolder, maxClusterSize);

    return { detail: 'Tree rebuilding completed' };
});

app.post('/maintain_tree', async (request, reply) => {
    console.log('\n\nMAINTAIN TREE');
    await utils.maintain_tree(dataFolder);

    return { detail: 'Tree maintenance completed' };
});

app.listen({
    port: 3000
}, (err, address) => {
    if (err) {
        app.log.error(err);
        process.exit(1);
    }
    app.log.info(`Server listening at ${address}`);
});

