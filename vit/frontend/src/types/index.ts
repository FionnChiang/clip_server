export interface Project {
  id: string;
  name: string;
  description: string;
  model_path: string;
  categories: string[];
  config: Record<string, any>;
  status: string;
  created_at: string;
  updated_at: string;
  image_count: number;
  model_count: number;
  latest_job_status: string | null;
}

export interface Category {
  name: string;
  image_count: number;
  train_count: number;
  val_count: number;
}

export interface DatasetImage {
  id: string;
  category: string;
  s3_url: string;
  original_filename: string;
  file_size: number;
  width: number;
  height: number;
  split: string;
  uploaded_at: string;
}

export interface PaginatedImages {
  items: DatasetImage[];
  total: number;
  page: number;
  page_size: number;
}

export interface TrainingJob {
  id: string;
  project_id: string;
  status: string;
  current_epoch: number;
  total_epochs: number;
  best_val_acc: number;
  started_at: string | null;
  finished_at: string | null;
}

export interface TrainingMetric {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
  lr: number;
}

export interface WSMessage {
  type: string;
  job_id: string;
  data: Record<string, any>;
}

export interface ModelInfo {
  id: string;
  project_id: string;
  job_id: string | null;
  name: string;
  checkpoint_path: string;
  val_acc: number;
  categories: string[];
  config: Record<string, any>;
  status: string;
  created_at: string | null;
}

export interface PredictionResult {
  category: string;
  index: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface MySQLConfig {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
}

export interface S3Config {
  endpoint: string;
  access_key: string;
  secret_key: string;
  bucket: string;
  region: string;
  use_ssl: boolean;
}

export interface SystemSettings {
  mysql: MySQLConfig;
  s3: S3Config;
}

export interface TrainingConfigPayload {
  training: {
    batch_size: number;
    epochs: number;
    lr: number;
    weight_decay: number;
    lr_scheduler: string;
    early_stop_patience: number;
    class_balance: string;
    num_workers: number;
    save_best_only: boolean;
    mixed_precision: boolean;
    log_interval: number;
  };
  model: {
    path: string;
    freeze_encoder: boolean;
    dropout: number;
    projection_dim: number | null;
    pool: string;
  };
  train_ratio: number;
  seed: number;
}
