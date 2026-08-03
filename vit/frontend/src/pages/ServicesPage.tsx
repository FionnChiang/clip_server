import { useOutletContext } from 'react-router-dom';
import { Card, Typography, Alert, Tag } from 'antd';
import type { Project } from '../types';

const { Text, Title } = Typography;

export default function ServicesPage() {
  const { project } = useOutletContext<{ project: Project; projectId: string }>();

  return (
    <div>
      <Card title="Service Deployment">
        <Alert
          message="Service deployment requires a trained model. Start the inference service from this interface."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Card title="Deployment Guide" size="small">
          <div style={{ lineHeight: 2 }}>
            <Text>1. Train a model in the <Tag color="blue">Model Training</Tag> tab.</Text><br />
            <Text>2. Deploy the trained model as an HTTP API using the "Start Service" button.</Text><br />
            <Text>3. The service exposes a REST API for image classification:</Text><br />
            <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4 }}>
{`POST /api/projects/{id}/predict
Content-Type: multipart/form-data

# Response:
{
  "category": "身份证",
  "index": 0,
  "confidence": 0.9567,
  "probabilities": {...}
}`}
            </pre>
            <Text>4. API docs available at <code>/docs</code> when the service is running.</Text><br />
            <Text>5. For dedicated inference, use the CLI command: </Text>
            <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4 }}>
              python scripts/serve.py --checkpoint &lt;checkpoint_path&gt; --port 8001
            </pre>
          </div>
        </Card>
      </Card>
    </div>
  );
}
