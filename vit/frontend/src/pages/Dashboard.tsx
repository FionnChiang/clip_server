import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Button, Modal, Form, Input, Empty, Spin, message, Popconfirm, Tag, Typography } from 'antd';
import { PlusOutlined, FolderOutlined, DeleteOutlined, SettingOutlined, ExperimentOutlined } from '@ant-design/icons';
import { listProjects, createProject, deleteProject } from '../api/projects';
import type { Project } from '../types';

const { Title, Text } = Typography;

export default function Dashboard() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [form] = Form.useForm();

  const load = async () => {
    try {
      setLoading(true);
      const data = await listProjects();
      setProjects(data);
    } catch {
      message.error('Failed to load projects');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleCreate = async () => {
    try {
      const values = await form.validateFields();
      setCreating(true);
      const p = await createProject(values);
      message.success('Project created');
      setModalOpen(false);
      form.resetFields();
      navigate(`/projects/${p.id}/dataset`);
    } catch {
      // validation error
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteProject(id);
      message.success('Project deleted');
      load();
    } catch {
      message.error('Failed to delete project');
    }
  };

  const statusColor: Record<string, string> = {
    active: 'green',
    archived: 'default',
    completed: 'blue',
    running: 'processing',
    failed: 'red',
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f5f5f5' }}>
      <div style={{ padding: '24px 32px', background: '#001529', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <ExperimentOutlined style={{ fontSize: 28, color: '#fff' }} />
          <Title level={4} style={{ color: '#fff', margin: 0 }}>Layout Classifier - 版式分类平台</Title>
        </div>
        <Button icon={<SettingOutlined />} onClick={() => navigate('/settings')} style={{ marginLeft: 16 }}>
          System Settings
        </Button>
      </div>

      <div style={{ maxWidth: 1200, margin: '24px auto', padding: '0 24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <Title level={3} style={{ margin: 0 }}>Projects</Title>
          <Button type="primary" icon={<PlusOutlined />} size="large" onClick={() => setModalOpen(true)}>
            New Project
          </Button>
        </div>

        <Spin spinning={loading}>
          {projects.length === 0 ? (
            <Card>
              <Empty description="No projects yet. Create one to get started.">
                <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
                  Create Project
                </Button>
              </Empty>
            </Card>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
              {projects.map((p) => (
                <Card
                  key={p.id}
                  hoverable
                  actions={[
                    <span key="imgs">{p.image_count} images</span>,
                    <span key="models">{p.model_count} models</span>,
                    <Popconfirm key="del" title="Delete this project?" onConfirm={() => handleDelete(p.id)}>
                      <DeleteOutlined style={{ color: '#ff4d4f' }} />
                    </Popconfirm>,
                  ]}
                  onClick={() => navigate(`/projects/${p.id}/dataset`)}
                >
                  <Card.Meta
                    avatar={<FolderOutlined style={{ fontSize: 32, color: '#1677ff' }} />}
                    title={p.name}
                    description={
                      <div>
                        <Text type="secondary" style={{ fontSize: 12 }}>{p.description || 'No description'}</Text>
                        <div style={{ marginTop: 8 }}>
                          {p.latest_job_status && (
                            <Tag color={statusColor[p.latest_job_status] || 'default'}>{p.latest_job_status}</Tag>
                          )}
                          {p.categories && p.categories.length > 0 && (
                            <Tag color="blue">{p.categories.length} categories</Tag>
                          )}
                        </div>
                      </div>
                    }
                  />
                </Card>
              ))}
            </div>
          )}
        </Spin>
      </div>

      <Modal
        title="Create New Project"
        open={modalOpen}
        onOk={handleCreate}
        onCancel={() => { setModalOpen(false); form.resetFields(); }}
        confirmLoading={creating}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="Project Name" rules={[{ required: true, message: 'Please input project name' }]}>
            <Input placeholder="e.g. Document Layout Classifier v2" />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea rows={3} placeholder="Optional description" />
          </Form.Item>
          <Form.Item name="model_path" label="CLIP Model Path" initialValue="../models">
            <Input placeholder="../models" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
