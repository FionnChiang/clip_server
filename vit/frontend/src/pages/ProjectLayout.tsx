import { useState, useEffect } from 'react';
import { Outlet, useParams, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Typography, Button, Spin, Breadcrumb } from 'antd';
import {
  PictureOutlined, ThunderboltOutlined, HddOutlined,
  ScanOutlined, CloudServerOutlined, ArrowLeftOutlined,
} from '@ant-design/icons';
import { getProject } from '../api/projects';
import type { Project } from '../types';

const { Sider, Content } = Layout;
const { Title } = Typography;

const menuItems = [
  { key: 'dataset', icon: <PictureOutlined />, label: 'Dataset Management' },
  { key: 'training', icon: <ThunderboltOutlined />, label: 'Model Training' },
  { key: 'models', icon: <HddOutlined />, label: 'Model Management' },
  { key: 'inference', icon: <ScanOutlined />, label: 'Inference Test' },
  { key: 'services', icon: <CloudServerOutlined />, label: 'Service Deployment' },
];

export default function ProjectLayoutPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);

  const currentTab = location.pathname.split('/').pop() || 'dataset';

  useEffect(() => {
    if (projectId) {
      setLoading(true);
      getProject(projectId)
        .then(setProject)
        .catch(() => navigate('/'))
        .finally(() => setLoading(false));
    }
  }, [projectId, navigate]);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!project) return null;

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={240} style={{ background: '#001529' }}>
        <div style={{ padding: '16px 24px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <Button
            type="link"
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate('/')}
            style={{ color: 'rgba(255,255,255,0.65)', padding: 0 }}
          >
            Back to Projects
          </Button>
          <Title level={5} style={{ color: '#fff', margin: '8px 0 0' }}>{project.name}</Title>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[currentTab]}
          items={menuItems}
          onClick={({ key }) => navigate(`/projects/${projectId}/${key}`)}
          style={{ marginTop: 8 }}
        />
      </Sider>
      <Layout>
        <Content style={{ padding: 24, background: '#f5f5f5', minHeight: '100vh' }}>
          <Breadcrumb
            style={{ marginBottom: 16 }}
            items={[
              { title: <a onClick={() => navigate('/')}>Projects</a> },
              { title: project.name },
              { title: menuItems.find((m) => m.key === currentTab)?.label },
            ]}
          />
          <Outlet context={{ project, projectId, onProjectUpdate: setProject }} />
        </Content>
      </Layout>
    </Layout>
  );
}
