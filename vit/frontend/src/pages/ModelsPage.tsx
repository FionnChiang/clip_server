import { useState, useEffect } from 'react';
import { useOutletContext } from 'react-router-dom';
import { Card, Table, Tag, Button, Popconfirm, message, Space, Typography, Descriptions } from 'antd';
import { DeleteOutlined, ReloadOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { listModels, deleteModel } from '../api/models';
import type { ModelInfo, Project } from '../types';

const { Text } = Typography;

export default function ModelsPage() {
  const { projectId } = useOutletContext<{ project: Project; projectId: string }>();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await listModels(projectId);
      setModels(data);
    } catch { message.error('Failed to load models'); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, [projectId]);

  const handleDelete = async (id: string) => {
    await deleteModel(projectId, id);
    message.success('Model deleted');
    load();
  };

  const columns: ColumnsType<ModelInfo> = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    {
      title: 'Val Accuracy', dataIndex: 'val_acc', key: 'val_acc',
      render: (v: number) => <Tag color="blue">{v?.toFixed(4) || '-'}</Tag>,
    },
    {
      title: 'Categories', dataIndex: 'categories', key: 'categories',
      render: (cats: string[]) => (
        <Space wrap>{cats.map((c) => <Tag key={c}>{c}</Tag>)}</Space>
      ),
    },
    {
      title: 'Created', dataIndex: 'created_at', key: 'created_at',
      render: (t: string) => t ? new Date(t).toLocaleString() : '-',
    },
    {
      title: 'Action', key: 'action',
      render: (_, record) => (
        <Space>
          <Button type="link" size="small" onClick={() => setSelectedModel(record)}>Detail</Button>
          <Popconfirm title="Delete this model?" onConfirm={() => handleDelete(record.id)}>
            <Button type="link" danger size="small" icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Card
        title="Trained Models"
        extra={<Button icon={<ReloadOutlined />} onClick={load}>Refresh</Button>}
      >
        <Table columns={columns} dataSource={models} rowKey="id" loading={loading} size="small" />
      </Card>

      {selectedModel && (
        <Card title="Model Detail" style={{ marginTop: 16 }}>
          <Descriptions column={2} bordered size="small">
            <Descriptions.Item label="Name">{selectedModel.name}</Descriptions.Item>
            <Descriptions.Item label="Val Accuracy">{selectedModel.val_acc?.toFixed(4)}</Descriptions.Item>
            <Descriptions.Item label="Categories">{selectedModel.categories?.join(', ') || '-'}</Descriptions.Item>
            <Descriptions.Item label="Checkpoint">{selectedModel.checkpoint_path || '-'}</Descriptions.Item>
            <Descriptions.Item label="Created">{selectedModel.created_at ? new Date(selectedModel.created_at).toLocaleString() : '-'}</Descriptions.Item>
          </Descriptions>
        </Card>
      )}
    </div>
  );
}
