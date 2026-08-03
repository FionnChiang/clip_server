import { useState, useEffect, useCallback } from 'react';
import { useOutletContext } from 'react-router-dom';
import {
  Card, Button, Upload, Tabs, Table, Tag, Space, message,
  Popconfirm, Modal, Input, Select, Row, Col, Statistic, Spin, Empty,
} from 'antd';
import {
  UploadOutlined, InboxOutlined, DeleteOutlined, ReloadOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { UploadProps } from 'antd';
import { listCategories, removeCategory, uploadImages, listImages, deleteImage, applySplit } from '../api/datasets';
import { getProject } from '../api/projects';
import type { Category, DatasetImage, Project } from '../types';

const { Dragger } = Upload;

export default function DatasetPage() {
  const { project, projectId, onProjectUpdate } = useOutletContext<{
    project: Project;
    projectId: string;
    onProjectUpdate: (p: Project) => void;
  }>();

  const [categories, setCategories] = useState<Category[]>([]);
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [activeCategory, setActiveCategory] = useState<string>('');
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadCategory, setUploadCategory] = useState('');
  const [uploading, setUploading] = useState(false);
  const [newCategoryName, setNewCategoryName] = useState('');
  const [splitRatio, setSplitRatio] = useState(0.8);

  const loadCategories = useCallback(async () => {
    const cats = await listCategories(projectId);
    setCategories(cats);
  }, [projectId]);

  const loadImages = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listImages(projectId, {
        category: activeCategory || undefined,
        page,
        page_size: 20,
      });
      setImages(data.items);
      setTotal(data.total);
    } catch {
      message.error('Failed to load images');
    } finally {
      setLoading(false);
    }
  }, [projectId, activeCategory, page]);

  useEffect(() => { loadCategories(); }, [loadCategories]);
  useEffect(() => { loadImages(); }, [loadImages]);

  const handleUpload = async () => {
    setUploading(true);
    try {
      const res = await uploadImages(projectId, uploadFiles, uploadCategory);
      message.success(`Uploaded ${res.uploaded} images`);
      setUploadOpen(false);
      setUploadFiles([]);
      loadCategories();
      loadImages();
    } catch (e: any) {
      message.error(e.message);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteCat = async (name: string) => {
    await removeCategory(projectId, name);
    message.success(`Category "${name}" deleted`);
    loadCategories();
    loadImages();
  };

  const handleSplit = async () => {
    await applySplit(projectId, { train_ratio: splitRatio, seed: 42 });
    message.success('Train/validation split applied');
    loadCategories();
  };

  const [uploadFiles, setUploadFiles] = useState<File[]>([]);

  const uploadProps: UploadProps = {
    multiple: true,
    beforeUpload: (file) => {
      setUploadFiles((prev) => [...prev, file]);
      return false;
    },
    onRemove: (file) => {
      setUploadFiles((prev) => prev.filter((f) => f.name !== file.name));
    },
  };

  const columns: ColumnsType<DatasetImage> = [
    {
      title: 'Image',
      dataIndex: 's3_url',
      key: 'preview',
      width: 80,
      render: (url: string) => (
        <img src={url} alt="" style={{ width: 48, height: 48, objectFit: 'cover', borderRadius: 4 }}
          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
        />
      ),
    },
    { title: 'Filename', dataIndex: 'original_filename', key: 'name', ellipsis: true },
    { title: 'Category', dataIndex: 'category', key: 'category', render: (c: string) => <Tag color="blue">{c}</Tag> },
    {
      title: 'Split', dataIndex: 'split', key: 'split',
      render: (s: string) => <Tag color={s === 'train' ? 'green' : s === 'val' ? 'orange' : 'default'}>{s}</Tag>,
    },
    {
      title: 'Size', dataIndex: 'file_size', key: 'size',
      render: (s: number) => `${(s / 1024).toFixed(1)} KB`,
    },
    {
      title: 'Action', key: 'action', render: (_, record) => (
        <Popconfirm title="Delete this image?" onConfirm={async () => {
          await deleteImage(projectId, record.id);
          loadImages();
          loadCategories();
        }}>
          <Button type="link" danger icon={<DeleteOutlined />} size="small" />
        </Popconfirm>
      ),
    },
  ];

  const totalImages = categories.reduce((sum, c) => sum + c.image_count, 0);

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}><Card><Statistic title="Total Images" value={totalImages} /></Card></Col>
        <Col span={6}><Card><Statistic title="Categories" value={categories.length} /></Card></Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Train / Val Split"
              value={`${splitRatio * 100} / ${(1 - splitRatio) * 100}`}
              suffix="%"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Space>
              <Button type="primary" icon={<UploadOutlined />} onClick={() => setUploadOpen(true)}>Upload</Button>
              <Button icon={<ReloadOutlined />} onClick={() => { loadCategories(); loadImages(); }} />
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="Categories" style={{ marginBottom: 16 }}>
        <Tabs
          activeKey={activeCategory}
          onChange={(key) => { setActiveCategory(key); setPage(1); }}
          type="card"
          tabBarExtraContent={
            <Space style={{ marginRight: 16 }}>
              <Input
                size="small"
                placeholder="New category name"
                value={newCategoryName}
                onChange={(e) => setNewCategoryName(e.target.value)}
                style={{ width: 150 }}
              />
              <Button
                size="small"
                onClick={() => {
                  if (newCategoryName.trim()) {
                    setUploadCategory(newCategoryName.trim());
                    setUploadOpen(true);
                    setNewCategoryName('');
                  }
                }}
              >
                + Add Category
              </Button>
            </Space>
          }
          items={[
            { key: '', label: `All (${totalImages})` },
            ...categories.map((c) => ({
              key: c.name,
              label: (
                <span>
                  {c.name} ({c.image_count})
                  <Popconfirm title={`Delete "${c.name}" and all its images?`} onConfirm={() => handleDeleteCat(c.name)}>
                    <DeleteOutlined style={{ marginLeft: 6, color: '#ff4d4f', fontSize: 12 }} />
                  </Popconfirm>
                </span>
              ),
            })),
          ]}
        />
      </Card>

      <Card
        title="Split Configuration"
        style={{ marginBottom: 16 }}
        extra={<Button onClick={handleSplit}>Apply Split</Button>}
      >
        <Space>
          <span>Training ratio:</span>
          <Input
            type="number"
            min={0.1}
            max={0.9}
            step={0.05}
            value={splitRatio}
            onChange={(e) => setSplitRatio(Number(e.target.value))}
            style={{ width: 80 }}
          />
          <span>Validation ratio: {1 - splitRatio}</span>
        </Space>
      </Card>

      <Card title="Images">
        <Table
          columns={columns}
          dataSource={images}
          rowKey="id"
          loading={loading}
          pagination={{
            current: page,
            total,
            pageSize: 20,
            onChange: (p) => setPage(p),
            showTotal: (t) => `${t} images`,
          }}
          size="small"
        />
      </Card>

      <Modal
        title="Upload Images"
        open={uploadOpen}
        onOk={handleUpload}
        onCancel={() => { setUploadOpen(false); setUploadFiles([]); }}
        confirmLoading={uploading}
        width={600}
      >
        <div style={{ marginBottom: 16 }}>
          <span>Category: </span>
          <Input
            value={uploadCategory}
            onChange={(e) => setUploadCategory(e.target.value)}
            placeholder="Select or type category name"
            style={{ width: 300 }}
          />
        </div>
        <Dragger {...uploadProps} fileList={uploadFiles.map((f: any) => ({ ...f, uid: f.name, name: f.name }))}>
          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
          <p className="ant-upload-text">Click or drag images to upload</p>
          <p className="ant-upload-hint">Supports JPG, PNG, BMP, TIFF, WebP</p>
        </Dragger>
      </Modal>
    </div>
  );
}
