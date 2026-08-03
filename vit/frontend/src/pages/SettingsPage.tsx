import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card, Form, Input, InputNumber, Switch, Button, message, Space, Alert, Typography, Divider, Spin,
} from 'antd';
import { ArrowLeftOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import { getSettings, updateMySQL, testMySQL, updateS3, testS3 } from '../api/settings';

const { Title, Text } = Typography;

export default function SettingsPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [mysqlForm] = Form.useForm();
  const [s3Form] = Form.useForm();
  const [mysqlStatus, setMysqlStatus] = useState<{ ok: boolean; message: string } | null>(null);
  const [s3Status, setS3Status] = useState<{ ok: boolean; message: string } | null>(null);

  useEffect(() => {
    getSettings().then((data) => {
      mysqlForm.setFieldsValue(data.mysql);
      s3Form.setFieldsValue(data.s3);
    }).catch(() => {});
  }, [mysqlForm, s3Form]);

  const handleMysqlSave = async () => {
    try {
      const values = await mysqlForm.validateFields();
      setLoading(true);
      await updateMySQL(values);
      message.success('MySQL config saved');
    } catch {
      // validation
    } finally {
      setLoading(false);
    }
  };

  const handleMysqlTest = async () => {
    try {
      const values = mysqlForm.getFieldsValue();
      const result = await testMySQL(values);
      setMysqlStatus(result);
    } catch {
      setMysqlStatus({ ok: false, message: 'Validation failed' });
    }
  };

  const handleS3Save = async () => {
    try {
      const values = await s3Form.validateFields();
      setLoading(true);
      await updateS3(values);
      message.success('S3 config saved');
    } catch {
      // validation
    } finally {
      setLoading(false);
    }
  };

  const handleS3Test = async () => {
    try {
      const result = await testS3();
      setS3Status(result);
    } catch {
      setS3Status({ ok: false, message: 'Connection test failed' });
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f5f5f5', padding: 24 }}>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/')}>Back</Button>
          <Title level={3} style={{ margin: 0 }}>System Settings</Title>
        </div>

        <Card title="MySQL Configuration" style={{ marginBottom: 24 }}>
          <Form form={mysqlForm} layout="vertical">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Form.Item name="host" label="Host" rules={[{ required: true }]}>
                <Input placeholder="127.0.0.1" />
              </Form.Item>
              <Form.Item name="port" label="Port" rules={[{ required: true }]}>
                <InputNumber min={1} max={65535} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item name="user" label="User" rules={[{ required: true }]}>
                <Input placeholder="root" />
              </Form.Item>
              <Form.Item name="password" label="Password">
                <Input.Password placeholder="Password" />
              </Form.Item>
              <Form.Item name="database" label="Database" rules={[{ required: true }]}>
                <Input placeholder="layout_classifier" />
              </Form.Item>
            </Space>
            <Space>
              <Button type="primary" onClick={handleMysqlSave}>Save</Button>
              <Button onClick={handleMysqlTest}>Test Connection</Button>
            </Space>
          </Form>
          {mysqlStatus && (
            <Alert
              style={{ marginTop: 12 }}
              type={mysqlStatus.ok ? 'success' : 'error'}
              message={mysqlStatus.message}
              showIcon
              icon={mysqlStatus.ok ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
            />
          )}
        </Card>

        <Card title="S3 Configuration">
          <Form form={s3Form} layout="vertical">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Form.Item name="endpoint" label="Endpoint URL">
                <Input placeholder="https://s3.amazonaws.com (leave empty for AWS S3)" />
              </Form.Item>
              <Form.Item name="access_key" label="Access Key" rules={[{ required: true }]}>
                <Input placeholder="Access Key ID" />
              </Form.Item>
              <Form.Item name="secret_key" label="Secret Key" rules={[{ required: true }]}>
                <Input.Password placeholder="Secret Access Key" />
              </Form.Item>
              <Form.Item name="bucket" label="Bucket" rules={[{ required: true }]}>
                <Input placeholder="layout-classifier" />
              </Form.Item>
              <Form.Item name="region" label="Region">
                <Input placeholder="us-east-1" />
              </Form.Item>
              <Form.Item name="use_ssl" label="Use SSL" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Space>
            <Space>
              <Button type="primary" onClick={handleS3Save}>Save</Button>
              <Button onClick={handleS3Test}>Test Connection</Button>
            </Space>
          </Form>
          {s3Status && (
            <Alert
              style={{ marginTop: 12 }}
              type={s3Status.ok ? 'success' : 'error'}
              message={s3Status.message}
              showIcon
              icon={s3Status.ok ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
            />
          )}
        </Card>
      </div>
    </div>
  );
}
