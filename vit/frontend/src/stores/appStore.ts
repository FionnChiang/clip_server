import { create } from 'zustand';

interface AppState {
  currentProjectId: string | null;
  setCurrentProjectId: (id: string | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentProjectId: null,
  setCurrentProjectId: (id) => set({ currentProjectId: id }),
}));
