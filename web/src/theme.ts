import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#e53935",
    },
    background: {
      default: "#121212",
      paper: "#1e1e1e",
    },
  },
});

export default theme;
