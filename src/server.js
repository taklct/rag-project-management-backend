import express from 'express';
import cors from 'cors';
import multer from 'multer';
import stringSimilarity from 'string-similarity';
import XLSX from 'xlsx';

const app = express();
const port = process.env.PORT || 3000;

const upload = multer({ storage: multer.memoryStorage() });

const knowledgeBase = [];

const allowedMimeTypes = new Set([
  'application/vnd.ms-excel',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
]);

const columnMap = {
  'Sprint Number': 'sprintNumber',
  Assignee: 'assignee',
  Team: 'team',
  'Task Title': 'taskTitle',
  'Task Description': 'taskDescription',
  'Task start date': 'taskStartDate',
  'Task end date': 'taskEndDate',
  Status: 'status',
  Priority: 'priority',
  'Story Point': 'storyPoint'
};

app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.get('/', (req, res) => {
  res.json({ message: 'Welcome to the RAG Project Management backend service.' });
});

app.post('/ingest', upload.single('file'), (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded. Expecting field name "file".' });
    }

    if (!allowedMimeTypes.has(req.file.mimetype)) {
      return res.status(400).json({ error: 'Unsupported file type. Please upload an Excel .xls or .xlsx file.' });
    }

    const workbook = XLSX.read(req.file.buffer, { type: 'buffer' });
    const firstSheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[firstSheetName];

    if (!worksheet) {
      return res.status(400).json({ error: 'Unable to read worksheet from the uploaded file.' });
    }

    const rows = XLSX.utils.sheet_to_json(worksheet, { defval: '' });

    if (!rows.length) {
      return res.status(400).json({ error: 'The uploaded worksheet does not contain any data.' });
    }

    knowledgeBase.length = 0;

    rows.forEach((row, index) => {
      const normalized = Object.entries(columnMap).reduce((acc, [column, key]) => {
        acc[key] = normalizeCell(row[column]);
        return acc;
      }, {});

      const content = buildDocumentContent(normalized);

      knowledgeBase.push({
        id: index + 1,
        ...normalized,
        content
      });
    });

    res.json({
      message: 'Excel file ingested successfully.',
      records: knowledgeBase.length
    });
  } catch (error) {
    next(error);
  }
});

app.post('/query', (req, res) => {
  const { question, topK = 3 } = req.body || {};

  if (!question || typeof question !== 'string') {
    return res.status(400).json({ error: 'The request body must include a "question" field.' });
  }

  if (!knowledgeBase.length) {
    return res.status(400).json({ error: 'Knowledge base is empty. Please ingest an Excel file first.' });
  }

  const cleanedQuestion = question.trim();

  const ranked = knowledgeBase
    .map((entry) => ({
      entry,
      score: scoreSimilarity(cleanedQuestion, entry.content)
    }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, Math.min(topK, knowledgeBase.length)));

  if (!ranked.length) {
    return res.json({
      answer: 'No relevant tasks were found for the provided question.',
      context: []
    });
  }

  const context = ranked.map(({ entry, score }) => ({
    id: entry.id,
    sprintNumber: entry.sprintNumber,
    assignee: entry.assignee,
    team: entry.team,
    taskTitle: entry.taskTitle,
    taskDescription: entry.taskDescription,
    taskStartDate: entry.taskStartDate,
    taskEndDate: entry.taskEndDate,
    status: entry.status,
    priority: entry.priority,
    storyPoint: entry.storyPoint,
    relevance: Number(score.toFixed(4))
  }));

  const bestMatch = context[0];
  const answer = buildAnswer(bestMatch);

  res.json({
    answer,
    context
  });
});

app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

function normalizeCell(value) {
  if (value === undefined || value === null) {
    return '';
  }

  if (value instanceof Date) {
    return value.toISOString().split('T')[0];
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toString();
  }

  return String(value).trim();
}

function buildDocumentContent(task) {
  return [
    `Sprint ${task.sprintNumber}`,
    `Assignee ${task.assignee}`,
    `Team ${task.team}`,
    `Task ${task.taskTitle}`,
    task.taskDescription,
    `Start ${task.taskStartDate}`,
    `End ${task.taskEndDate}`,
    `Status ${task.status}`,
    `Priority ${task.priority}`,
    `Story Points ${task.storyPoint}`
  ]
    .filter(Boolean)
    .join('. ');
}

function scoreSimilarity(question, content) {
  if (!content) {
    return 0;
  }

  const comparison = stringSimilarity.compareTwoStrings(question.toLowerCase(), content.toLowerCase());
  return Number.isFinite(comparison) ? comparison : 0;
}

function buildAnswer(task) {
  if (!task) {
    return 'No relevant tasks were found for the provided question.';
  }

  const details = [
    `Sprint ${task.sprintNumber}`,
    `Assignee ${task.assignee}`,
    `Team ${task.team}`,
    `Title: ${task.taskTitle}`,
    `Status: ${task.status}`,
    `Priority: ${task.priority}`,
    `Timeline: ${task.taskStartDate} to ${task.taskEndDate}`,
    `Story Points: ${task.storyPoint}`
  ]
    .filter(Boolean)
    .join('; ');

  return `Most relevant task -> ${details}`;
}
