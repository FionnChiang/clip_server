import { useState, useEffect, useCallback, useRef } from 'react';
import { useOutletContext } from 'react-router-dom';
import {
  Card, Button, Form, InputNumber, Select, Switch, Slider, Space,
  Table, Tag, message, Row, Col, Typography, Progress, Alert,
} from 'antd';
import { ThunderboltOutlined, StopOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { listJobs, startTraining, stopTraining, getJobMetrics } from '../api/training';
import { useWebSocket } from '../hooks/useWebSocket';
import type { TrainingJob, TrainingMetric, Project, WSMessage } from '../types';

const { Title, Text } = Typography;
const { Option } = Select;

export default function TrainingPage() {
  const { project, projectId } = useOutletContext<{
    project: Project;
    projectId: string;
  }>();

  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [trainingStatus, setTrainingStatus] = useState<string>('');
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [training, setTraining] = useState(false);

  const handleWSMessage = useCallback((msg: WSMessage) => {
    if (msg.type === 'progress') {
      const d = msg.data;
      if (d.epoch) setCurrentEpoch(d.epoch);
      if (d.status) setTrainingStatus(d.status);
      if (d.train_loss !== undefined) {
        setMetrics((prev) => {
          const idx = prev.findIndex((m) => m.epoch === d.epoch);
          const newMetric: TrainingMetric = {
            epoch: d.epoch,
            train_loss: d.train_loss,
            train_acc: d.train_acc,
            val_loss: d.val_loss,
            val_acc: d.val_acc,
            lr: d.lr,
          };
          if (idx >= 0) {
            const copy = [...prev];
            copy[idx] = newMetric;
            return copy;
          }
          return [...prev, newMetric];
        });
      }
      if (d.status === 'early_stopping') {
        message.info(d.message || 'Early stopping triggered');
      }
    } else if (msg.type === 'completed') {
      message.success(`Training completed! Best val_acc: ${msg.data.best_val_acc?.toFixed(4)}`);
      setTraining(false);
      setTrainingStatus('completed');
      loadJobs();
    } else if (msg.type === 'error') {
      message.error(`Training failed: ${msg.data.message}`);
      setTraining(false);
      setTrainingStatus('failed');
      loadJobs();
    } else if (msg.type === 'stopped') {
      message.info('Training stopped');
      setTraining(false);
      setTrainingStatus('stopped');
      loadJobs();
    }
  }, []);

  useWebSocket(activeJobId, handleWSMessage);

  const loadJobs = useCallback(async () => {
    try {
      const data = await listJobs(projectId);
      setJobs(data);
      const running = data.find((j) => j.status === 'running');
      if (running) {
        setActiveJobId(running.id);
        setTraining(true);
        setTrainingStatus('running');
        setTotalEpochs(running.total_epochs);
        setCurrentEpoch(running.current_epoch);
      }
    } catch {
      message.error('Failed to load jobs');
    }
  }, [projectId]);

  useEffect(() => { loadJobs(); }, [loadJobs]);

  const onFinish = async (values: any) => {
    const payload = {
      training: {
        batch_size: values.batch_size,
        epochs: values.epochs,
        lr: values.lr,
        weight_decay: values.weight_decay,
        lr_scheduler: values.lr_scheduler,
        early_stop_patience: values.early_stop_patience,
        class_balance: values.class_balance,
        num_workers: values.num_workers,
        save_best_only: values.save_best_only,
        mixed_precision: values.mixed_precision,
        log_interval: values.log_interval,
      },
      model: {
        path: values.model_path || '../models',
        freeze_encoder: values.freeze_encoder,
        dropout: values.dropout,
        projection_dim: values.projection_dim || null,
        pool: values.pool,
      },
      train_ratio: values.train_ratio,
      seed: values.seed,
    };

    try {
      setTraining(true);
      setMetrics([]);
      setTrainingStatus('preparing');
      const job = await startTraining(projectId, payload);
      setActiveJobId(job.id);
      setTotalEpochs(job.total_epochs);
      setCurrentEpoch(0);
      message.info('Training started');
      loadJobs();
    } catch (e: any) {
      message.error(e.message);
      setTraining(false);
    }
  };

  const handleStop = async () => {
    if (!activeJobId) return;
    await stopTraining(projectId, activeJobId);
    setTraining(false);
  };

  const lossChartOption = {
    tooltip: { trigger: 'axis' },
    legend: { data: ['Train Loss', 'Val Loss'] },
    xAxis: { type: 'category', data: metrics.map((m) => `E${m.epoch}`) },
    yAxis: { type: 'value', name: 'Loss' },
    series: [
      { name: 'Train Loss', type: 'line', data: metrics.map((m) => m.train_loss), smooth: true, lineStyle: { color: '#1677ff' } },
      { name: 'Val Loss', type: 'line', data: metrics.map((m) => m.val_loss), smooth: true, lineStyle: { color: '#fa541c' } },
    ],
  };

  const accChartOption = {
    tooltip: { trigger: 'axis' },
    legend: { data: ['Train Acc', 'Val Acc'] },
    xAxis: { type: 'category', data: metrics.map((m) => `E${m.epoch}`) },
    yAxis: { type: 'value', name: 'Accuracy', min: 0, max: 1 },
    series: [
      { name: 'Train Acc', type: 'line', data: metrics.map((m) => m.train_acc), smooth: true, lineStyle: { color: '#1677ff' } },
      { name: 'Val Acc', type: 'line', data: metrics.map((m) => m.val_acc), smooth: true, lineStyle: { color: '#52c41a' } },
    ],
  };

  const jobColumns = [
    { title: 'Job ID', dataIndex: 'id', key: 'id', width: 100, render: (id: string) => id.slice(0, 8) },
    {
      title: 'Status', dataIndex: 'status', key: 'status',
      render: (s: string) => {
        const colors: Record<string, string> = { pending: 'default', running: 'processing', completed: 'success', failed: 'error', stopped: 'warning' };
        return <Tag color={colors[s] || 'default'}>{s}</Tag>;
      },
    },
    { title: 'Epoch', key: 'epoch', render: (_: any, r: TrainingJob) => `${r.current_epoch}/${r.total_epochs}` },
    { title: 'Best Val Acc', dataIndex: 'best_val_acc', key: 'best_val_acc', render: (v: number) => v?.toFixed(4) || '-' },
  ];

  return (
    <div>
      <Row gutter={24} style={{ marginBottom: 24 }}>
        <Col span={10}>
          <Card title={<><ThunderboltOutlined /> Training Configuration</>} size="small">
            <Form
              layout="vertical"
              onFinish={onFinish}
              initialValues={{
                batch_size: 32, epochs: 50, lr: 0.001, weight_decay: 0.0001,
                lr_scheduler: 'cosine', early_stop_patience: 10, class_balance: 'weighted_loss',
                num_workers: 4, save_best_only: true, mixed_precision: false, log_interval: 10,
                train_ratio: 0.8, seed: 42,
                model_path: '../models', freeze_encoder: true, dropout: 0.1, pool: 'cls',
              }}
              size="small"
            >
              <Row gutter={12}>
                <Col span={8}>
                  <Form.Item label="Batch Size" name="batch_size">
                    <InputNumber min={1} max={256} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="Epochs" name="epochs">
                    <InputNumber min={1} max={500} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="LR" name="lr">
                    <InputNumber min={0.00001} max={1} step={0.0001} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="Weight Decay" name="weight_decay">
                    <InputNumber min={0} max={1} step={0.0001} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="LR Scheduler" name="lr_scheduler">
                    <Select><Option value="cosine">Cosine</Option><Option value="step">Step</Option></Select>
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="Early Stop Patience" name="early_stop_patience">
                    <InputNumber min={1} max={100} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="Class Balance" name="class_balance">
                    <Select>
                      <Option value="none">None</Option>
                      <Option value="weighted_loss">Weighted Loss</Option>
                      <Option value="oversample">Oversample</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="Dropout" name="dropout">
                    <InputNumber min={0} max={0.9} step={0.05} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="Pool" name="pool">
                    <Select><Option value="cls">CLS</Option><Option value="mean">Mean</Option></Select>
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="Train Ratio" name="train_ratio">
                    <InputNumber min={0.5} max={0.95} step={0.05} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="Freeze Encoder" name="freeze_encoder" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="Mixed Precision" name="mixed_precision" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item>
                <Space>
                  <Button type="primary" htmlType="submit" icon={<ThunderboltOutlined />} loading={training} disabled={training}>
                    Start Training
                  </Button>
                  {training && (
                    <Button danger icon={<StopOutlined />} onClick={handleStop}>Stop</Button>
                  )}
                </Space>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col span={14}>
          {training ? (
            <Card title="Training Progress" size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text>Status: </Text>
                  <Tag color={trainingStatus === 'running' ? 'processing' : 'default'}>{trainingStatus}</Tag>
                  <Text> Epoch: {currentEpoch} / {totalEpochs}</Text>
                </div>
                <Progress
                  percent={totalEpochs > 0 ? Math.round((currentEpoch / totalEpochs) * 100) : 0}
                  status={trainingStatus === 'completed' ? 'success' : trainingStatus === 'failed' ? 'exception' : 'active'}
                />
                {metrics.length > 0 && (
                  <>
                    <ReactECharts option={lossChartOption} style={{ height: 200 }} />
                    <ReactECharts option={accChartOption} style={{ height: 200 }} />
                  </>
                )}
              </Space>
            </Card>
          ) : (
            <Card>
              {project.categories.length === 0 ? (
                <Alert message="No categories yet. Please upload images in Dataset Management first." type="warning" showIcon />
              ) : (
                <Alert message="Configure training parameters and click 'Start Training' to begin." type="info" showIcon />
              )}
            </Card>
          )}
        </Col>
      </Row>

      <Card title="Training History" size="small">
        <Table columns={jobColumns} dataSource={jobs} rowKey="id" size="small" pagination={false} />
      </Card>
    </div>
  );
}
