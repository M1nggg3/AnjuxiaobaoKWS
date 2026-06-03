package cn.org.wenet.wekws;

import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class CaptureContractTest {
    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void acceptsStableCaptureIdentifiers() {
        assertTrue(CaptureContract.isValidCaptureId("m1_2m_positive_0001_data_aishell_S0003_pos_01"));
        assertFalse(CaptureContract.isValidCaptureId("../escape"));
        assertFalse(CaptureContract.isValidCaptureId("contains space"));
    }

    @Test
    public void usesCaptureIdForAllArtifactNames() {
        CaptureContract.Files files = CaptureContract.filesFor("m1_1m_positive_0001", null);

        assertEquals("m1_1m_positive_0001_raw_16k_s16le.pcm", files.rawPcm.getName());
        assertEquals("m1_1m_positive_0001_enhanced_16k_s16le.pcm", files.enhancedPcm.getName());
        assertEquals("m1_1m_positive_0001.log", files.log.getName());
        assertEquals("m1_1m_positive_0001.json", files.status.getName());
    }

    @Test
    public void treatsAnExistingStatusFileAsRecoverableData() throws IOException {
        CaptureContract.Files files = CaptureContract.filesFor("m1_1m_positive_0002", folder.getRoot());
        assertTrue(files.ensureDirectories());
        assertTrue(files.status.createNewFile());

        assertTrue(CaptureContract.hasRecoverableStatus(files));
    }
}
