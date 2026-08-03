import { useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import {
  Card, Upload, Button, Select, Space, message, Typography, Progress, Tag, Table, Row, Col,
} from 'antd';
import { UploadOutlined, InboxOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { predict, predictTopK } from '../api/inference';
import { listModels } from '../api/models';
import type { PredictionResult, ModelInfo, Project } from '../types';

const { Dragger } = Upload;
const { Title, Text } = Typography;
const { Option } = Select;

export default function InferencePage() {
  const { project, projectId } = useOutletContext<{ project: Project; projectId: string }>();
  const [modelId, setModelId] = useState<string | undefined>();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [topKResults, setTopKResults] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string>('');

  const loadModels = async () => {
    try {
      const data = await listModels(projectId);
      setModels(data);
      if (data.length > 0 && !modelId) setModelId(data[0].id);
    } catch { message.error('Failed to load models'); }
  };

  const props: UploadProps = {
    multiple: false,
    showUploadList: false,
    beforeUpload: async (file) => {
      setLoading(true);
      setPreviewUrl(URL.createObjectURL(file));
      try {
        const res = await predict(projectId, file, modelId);
        setResult(res);
        const topK = await predictTopK(projectId, file, 3, modelId);
        setTopKResults(topK.results || []);
        message.success('Prediction complete');
      } catch (e: any) {
        message.error(e.message);
      } finally {
        setLoading(false);
      }
      return false;
    },
  };

  return (
    <div>
      <Row gutter={24}>
        <Col span={10}>
          <Card title="Test Inference" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text>Model: </Text>
                <Select
                  value={modelId}
                  onChange={setModelId}
                  style={{ width: '100%' }}
                  onFocus={loadModels}
                  placeholder="Select a model"
                >
                  {models.map((m) => (
                    <Option key={m.id} value={m.id}>
                      {m.name} (acc: {m.val_acc?.toFixed(4)})
                    </Option>
                  ))}
                </Select>
              </div>
              <Dragger {...props} style={{ padding: 16 }}>
                <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                <p className="ant-upload-text">Click or drag an image for prediction</p>
              </Dragger>
              {previewUrl && (
                <img src={previewUrl} alt="preview" style={{ width: '100%', maxHeight: 300, objectFit: 'contain', borderRadius: 4 }} />
              )}
            </Space>
          </Card>
        </Col>

        <Col span={14}>
          <Card title="Prediction Results" loading={loading} size="small">
            {result ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ textAlign: 'center' }}>
                  <Title level={2} style={{ color: '#1677ff', margin: 0 }}>{result.category}</Title>
                  <Text type="secondary">Confidence: {(result.confidence * 100).toFixed(2)}%</Text>
                  <Progress percent={Math.round(result.confidence * 100)} style={{ marginTop: 8 }} />
                </div>

                <Card title="Top-K Results" size="small" style={{ marginTop: 16 }}>
                  <Table
                    dataSource={topKResults}
                    rowKey="index"
                    pagination={false}
                    size="small"
                    columns={[
                      { title: 'Category', dataIndex: 'category', key: 'category' },
                      {
                        title: 'Confidence', dataIndex: 'confidence', key: 'confidence',
                        render: (v: number) => `${(v * 100).toFixed(2)}%`,
                      },
                    ]}
                  />
                </Card>

                <Card title="All Probabilities" size="small">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {Object.entries(result.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([cat, prob]) => (
                        <div key={cat}>
                          <Space>
                            <Tag color={cat === result.category ? 'blue' : 'default'}>{cat}</Tag>
                            <Progress percent={Math.round(prob * 100)} size="small" style={{ width: 200 }} />
                            <Text>{(prob * 100).toFixed(2)}%</Text>
                          </Space>
                        </div>
                      ))}
                  </Space>
                </Card>
              </Space>
            ) : (
              <Text type="secondary">Upload an image to see prediction results</Text>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
