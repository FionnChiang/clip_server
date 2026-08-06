import { useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import {
  Card, Upload, Button, Select, Space, message, Typography, Progress, Tag, Table, Row, Col, Collapse,
} from 'antd';
import { UploadOutlined, InboxOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { predict, predictTopK, type PredictResponse, type PredictTopKResponse } from '../api/inference';
import { listModels } from '../api/models';
import type { PredictionResult, ModelInfo, Project, DocumentPrediction, DocumentTopK } from '../types';

const { Dragger } = Upload;
const { Title, Text } = Typography;
const { Option } = Select;

const isDocumentFile = (name: string) => /\.(pdf|ofd)$/i.test(name);
const isDocumentResult = (res: PredictResponse): res is DocumentPrediction => 'page_count' in res;

const REASON_LABELS: Record<string, string> = {
  low_confidence: 'Low confidence',
  ambiguous: 'Ambiguous (top classes too close)',
};

const reasonTag = (reason: string | null) =>
  reason ? <Tag color="orange">{REASON_LABELS[reason] || reason}</Tag> : null;

export default function InferencePage() {
  const { project, projectId } = useOutletContext<{ project: Project; projectId: string }>();
  const [modelId, setModelId] = useState<string | undefined>();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [topKResults, setTopKResults] = useState<PredictionResult[]>([]);
  const [docTopK, setDocTopK] = useState<DocumentTopK | null>(null);
  const [loading, setLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');

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
      setFileName(file.name);
      try {
        const res = await predict(projectId, file, modelId);
        setResult(res);
        if (isDocumentResult(res)) {
          const topK = await predictTopK(projectId, file, 3, modelId);
          if ('page_count' in topK) setDocTopK(topK);
        } else {
          const topK = await predictTopK(projectId, file, 3, modelId);
          if ('page_count' in topK) {
            setDocTopK(topK);
          } else {
            setTopKResults(topK.results || []);
          }
        }
        message.success('Prediction complete');
      } catch (e: any) {
        message.error(e.message);
      } finally {
        setLoading(false);
      }
      return false;
    },
  };

  const renderProbabilities = (probabilities: Record<string, number>, topCategory: string) => (
    <Space direction="vertical" style={{ width: '100%' }}>
      {Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([cat, prob]) => (
          <div key={cat}>
            <Space>
              <Tag color={cat === topCategory ? 'blue' : 'default'}>{cat}</Tag>
              <Progress percent={Math.round(prob * 100)} size="small" style={{ width: 200 }} />
              <Text>{(prob * 100).toFixed(2)}%</Text>
            </Space>
          </div>
        ))}
    </Space>
  );

  const renderDocumentResult = (doc: DocumentPrediction) => (
    <Space direction="vertical" style={{ width: '100%' }}>
      <Text type="secondary">
        {doc.filename} · {doc.page_count} page{doc.page_count > 1 ? 's' : ''}
      </Text>
      <Collapse
        defaultActiveKey={[1]}
        items={doc.results.map((p) => {
          const pageTopK = docTopK?.results.find((t) => t.page === p.page)?.top_k ?? [];
          return {
            key: p.page,
            label: (
              <Space>
                <Text strong>Page {p.page}</Text>
                <Text style={{ color: p.rejected ? '#fa8c16' : '#1677ff' }}>{p.category}</Text>
                <Text type="secondary">{(p.confidence * 100).toFixed(2)}%</Text>
                {reasonTag(p.reason ? p.reason : null)}
              </Space>
            ),
            children: (
              <Space direction="vertical" style={{ width: '100%' }}>
                <Progress
                  percent={Math.round(p.confidence * 100)}
                  status={p.rejected ? 'exception' : 'normal'}
                />
                {p.rejected && (
                  <Text type="secondary">
                    Classified as other. Model&apos;s top choice: {p.original_category} ({(p.confidence * 100).toFixed(2)}%)
                  </Text>
                )}
                {pageTopK.length > 0 && (
                  <Table
                    dataSource={pageTopK}
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
                )}
                {renderProbabilities(p.probabilities, p.category)}
              </Space>
            ),
          };
        })}
      />
    </Space>
  );

  const renderImageResult = (res: PredictionResult) => (
    <Space direction="vertical" style={{ width: '100%' }}>
      <div style={{ textAlign: 'center' }}>
        <Title level={2} style={{ color: res.rejected ? '#fa8c16' : '#1677ff', margin: 0 }}>
          {res.category}
        </Title>
        {res.rejected && reasonTag(res.reason)}
        <Text type="secondary">
          Confidence: {(res.confidence * 100).toFixed(2)}%
        </Text>
        <Progress
          percent={Math.round(res.confidence * 100)}
          status={res.rejected ? 'exception' : 'normal'}
          style={{ marginTop: 8 }}
        />
        {res.rejected && (
          <div style={{ marginTop: 8 }}>
            <Text type="secondary">
              Classified as other. Model&apos;s top choice: <Text strong>{res.original_category}</Text> ({(res.confidence * 100).toFixed(2)}%)
            </Text>
          </div>
        )}
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
        {renderProbabilities(res.probabilities, res.category)}
      </Card>
    </Space>
  );

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
                <p className="ant-upload-text">Click or drag an image or PDF/OFD document for prediction</p>
              </Dragger>
              {fileName && isDocumentFile(fileName) ? (
                <Tag icon={<UploadOutlined />} color="geekblue">{fileName}</Tag>
              ) : (
                previewUrl && (
                  <img src={previewUrl} alt="preview" style={{ width: '100%', maxHeight: 300, objectFit: 'contain', borderRadius: 4 }} />
                )
              )}
            </Space>
          </Card>
        </Col>

        <Col span={14}>
          <Card title="Prediction Results" loading={loading} size="small">
            {result ? (
              isDocumentResult(result) ? renderDocumentResult(result) : renderImageResult(result)
            ) : (
              <Text type="secondary">Upload an image or PDF/OFD document to see prediction results</Text>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
